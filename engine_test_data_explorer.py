import os
import sys
import traceback
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QListWidget, QListWidgetItem, QGroupBox,
    QComboBox, QTableWidget, QTableWidgetItem, QAbstractItemView, QRadioButton,
    QCheckBox, QStatusBar, QAction, QToolBar, QFileDialog, QProgressBar,
    QTabWidget, QSizePolicy, QMessageBox, QDialog, QMenu, QTableView, QHeaderView, QDateTimeEdit
)
from PyQt5.QtCore import Qt, QSettings, QSize, QTimer, QThread, pyqtSignal, QDateTime, QEvent
from PyQt5.QtGui import QIcon, QCursor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==== File Discovery (Recursive) ====
def find_data_files(root_folder, exts=('.txt', '.tsv')):
    result = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.lower().endswith(exts):
                result.append(os.path.join(dirpath, fname))
    return result

class FolderScanThread(QThread):
    finished = pyqtSignal(list)
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
    def run(self):
        files = find_data_files(self.folder)
        self.finished.emit(files)

# ==== Header Preview Dialog ====
class HeaderPreviewDialog(QDialog):
    def __init__(self, filepaths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview File Headers")
        layout = QVBoxLayout(self)
        tabs = QTabWidget(self)
        decimal_sep = self.decimal_sep_combo.currentText()
        for filepath in filepaths:
            tab = QWidget()
            vbox = QVBoxLayout(tab)
            table = QTableWidget()
            vbox.addWidget(table)
            try:
                df = pd.read_csv(filepath, sep='\t', nrows=5, decimal=decimal_sep)
                table.setRowCount(len(df))
                table.setColumnCount(len(df.columns))
                table.setHorizontalHeaderLabels(list(df.columns))
                for i, row in df.iterrows():
                    for j, val in enumerate(row):
                        table.setItem(i, j, QTableWidgetItem(str(val)))
            except Exception as e:
                table.setRowCount(1)
                table.setColumnCount(1)
                table.setItem(0, 0, QTableWidgetItem(f"Failed: {e}"))
            tabs.addTab(tab, os.path.basename(filepath))
        layout.addWidget(tabs)
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

# ==== Data Utilities ====
def is_time_col(col):
    return any(x in col.lower() for x in ["time", "zaman", "date"])

def read_file_headers(files):
    headers = {}
    for f in files:
        try:
            df = pd.read_csv(f, sep='\t', nrows=1)
            headers[f] = list(df.columns)
        except Exception:
            headers[f] = []
    return headers

def unique_numeric_columns(files):
    cols = set()
    decimal_sep = self.decimal_sep_combo.currentText()
    for f in files:
        try:
            df = pd.read_csv(f, sep='\t', nrows=10, decimal=decimal_sep)
            for c in df.columns:
                try:
                    pd.to_numeric(df[c])
                    cols.add(c)
                except Exception:
                    continue
        except Exception:
            continue
    return sorted(list(cols))

def extract_datetime_col(df, time_col):
    try:
        t = pd.to_datetime(df[time_col])
        t0 = t.iloc[0]
        dt_sec = (t - t0).dt.total_seconds()
        return dt_sec
    except Exception:
        # Try convert to numeric as fallback
        return pd.to_numeric(df[time_col], errors="coerce")

# ==== Main Window ====
class EngineTestDataExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Engine Test Data Explorer")
        self.setMinimumSize(1280, 750)
        self.settings = QSettings("ChatGPT", "EngineTestDataExplorer")
        self.file_list_data = []
        self.filtered_files = []
        self.scanning = False
        self.loaded_dfs = {}  # filename: DataFrame
        self.axis_rows = []   # for dynamic axes
        self.axis_map = []    # [(axis_name, column_name)]

        # Central Layout
        central_widget = QWidget(self)
        central_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        self.file_splitter = QSplitter(Qt.Horizontal)
        central_layout.addWidget(self.file_splitter)
        self.file_pane = self.create_file_pane()
        self.file_splitter.addWidget(self.file_pane)
        self.right_splitter = QSplitter(Qt.Vertical)
        self.file_splitter.addWidget(self.right_splitter)
        self.controls_pane = self.create_controls_pane()
        self.right_splitter.addWidget(self.controls_pane)
        self.plot_pane = self.create_plot_pane()
        self.right_splitter.addWidget(self.plot_pane)
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.init_menus_and_toolbars()
        self.restore_window_settings()
        self.init_dynamic_axis_table()

        # Button/handler connections
        self.btn_open_folder.clicked.connect(self.open_folder)
        self.btn_reload_folder.clicked.connect(self.reload_folder)
        self.btn_select_all.clicked.connect(self.select_all_files)
        self.btn_deselect_all.clicked.connect(self.deselect_all_files)
        self.filter_edit.textChanged.connect(self.filter_files)
        self.btn_preview_headers.clicked.connect(self.preview_headers)
        self.btn_add_axis.clicked.connect(self.add_axis_row)
        self.radio_overlay.toggled.connect(self.axis_mode_changed)
        self.btn_plot.clicked.connect(self.plot_data)
        self.axis_table.cellChanged.connect(self.axis_table_cell_changed)
        self.btn_export_csv.clicked.connect(self.export_plot_csv)
        self.btn_export_png.clicked.connect(self.export_plot_png)
        self.btn_preview_summary.clicked.connect(self.preview_summary)
        self.btn_export_summary.clicked.connect(self.export_summary)
        self.combo_summary_type.currentIndexChanged.connect(self.summary_type_changed)
        self.file_list.itemChanged.connect(self.on_file_checked)

        # Advanced: right-click context menu for plot
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.show_plot_context_menu)

        # For advanced summary: custom multi-select dialog
        self.custom_metrics = ["mean", "median", "min", "max", "std"]
        self.custom_metrics_selected = ["mean"]

        # Persistent state restore
        QTimer.singleShot(300, self.restore_last_used_folder)

    # ============ File Pane ============
    def create_file_pane(self):
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(8, 8, 8, 8)
        files_group = QGroupBox("Files")
        vbox = QVBoxLayout(files_group)
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setReadOnly(True)
        self.folder_path_edit.setPlaceholderText("No folder selected...")
        vbox.addWidget(self.folder_path_edit)
        btnrow = QHBoxLayout()
        self.btn_open_folder = QPushButton("Open Folder…")
        self.btn_reload_folder = QPushButton("Reload")
        self.btn_select_all = QPushButton("Select All")
        self.btn_deselect_all = QPushButton("Deselect All")
        btnrow.addWidget(self.btn_open_folder)
        btnrow.addWidget(self.btn_reload_folder)
        btnrow.addWidget(self.btn_select_all)
        btnrow.addWidget(self.btn_deselect_all)
        vbox.addLayout(btnrow)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter files…")
        vbox.addWidget(self.filter_edit)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setMinimumHeight(200)
        vbox.addWidget(self.file_list)

        self.decimal_sep_combo = QComboBox()
        self.decimal_sep_combo.addItems([".", ","])
        self.decimal_sep_combo.setCurrentIndex(0)
        vbox.addWidget(QLabel("Decimal Separator:"))
        vbox.addWidget(self.decimal_sep_combo)

        layout.addWidget(files_group)
        return pane

    # ============ Controls Pane ============
    def create_controls_pane(self):
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(8, 8, 8, 0)
        # Axes & Options Group
        axes_group = QGroupBox("Axes & Options")
        agl = QVBoxLayout(axes_group)
        dt_row = QHBoxLayout()
        dt_row.addWidget(QLabel("Date/Time Column:"))
        self.combo_datetime = QComboBox()
        self.combo_datetime.setMinimumWidth(150)
        dt_row.addWidget(self.combo_datetime)
        agl.addLayout(dt_row)
        # Dynamic axis mapping table
        self.axis_table = QTableWidget(1, 3)
        self.axis_table.setHorizontalHeaderLabels(["Axis", "Column", "Remove"])
        self.axis_table.setVerticalHeaderLabels(["Primary Y"])
        self.axis_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        agl.addWidget(self.axis_table)
        axis_btnrow = QHBoxLayout()
        self.btn_add_axis = QPushButton("+ Add Axis")
        axis_btnrow.addWidget(self.btn_add_axis)
        agl.addLayout(axis_btnrow)
        mode_row = QHBoxLayout()
        self.radio_overlay = QRadioButton("Overlay (multiple axes)")
        self.radio_separate = QRadioButton("Separate Subplots")
        self.radio_overlay.setChecked(True)
        mode_row.addWidget(self.radio_overlay)
        mode_row.addWidget(self.radio_separate)
        agl.addLayout(mode_row)
        self.chk_autoscale = QCheckBox("Autoscale each axis")
        agl.addWidget(self.chk_autoscale)
        self.btn_plot = QPushButton("Plot")
        agl.addWidget(self.btn_plot)
        # Header preview
        self.btn_preview_headers = QPushButton("Preview Headers")
        agl.addWidget(self.btn_preview_headers)
        # Plot export
        exprow = QHBoxLayout()
        self.btn_export_csv = QPushButton("Export Plot CSV")
        self.btn_export_png = QPushButton("Export PNG")
        exprow.addWidget(self.btn_export_csv)
        exprow.addWidget(self.btn_export_png)
        agl.addLayout(exprow)
        layout.addWidget(axes_group)
        # Export Summaries Group
        summaries_group = QGroupBox("Export Summaries")
        sgl = QVBoxLayout(summaries_group)
        sum_row1 = QHBoxLayout()
        sum_row1.addWidget(QLabel("Summary Type:"))
        self.combo_summary_type = QComboBox()
        self.combo_summary_type.addItems(["Mean", "Median", "Min/Max", "Std Dev", "Custom…"])
        sum_row1.addWidget(self.combo_summary_type)
        sgl.addLayout(sum_row1)
        # Time window
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Time Window:"))
        self.start_time_edit = QLineEdit()
        self.start_time_edit.setPlaceholderText("Start datetime (opt)")
        self.end_time_edit = QLineEdit()
        self.end_time_edit.setPlaceholderText("End datetime (opt)")
        time_row.addWidget(self.start_time_edit)
        time_row.addWidget(self.end_time_edit)
        sgl.addLayout(time_row)
        # Buttons: Preview, Export
        sum_btnrow = QHBoxLayout()
        self.btn_preview_summary = QPushButton("Preview Summary")
        self.btn_export_summary = QPushButton("Export Summary…")
        sum_btnrow.addWidget(self.btn_preview_summary)
        sum_btnrow.addWidget(self.btn_export_summary)
        sgl.addLayout(sum_btnrow)
        layout.addWidget(summaries_group)
        layout.addStretch(1)
        return pane

    # ============ Plot Pane ============
    def create_plot_pane(self):
        plot_group = QGroupBox("Plot")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        return plot_group

    # ============ Menu, Toolbar, Shortcuts ============
    def init_menus_and_toolbars(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        act_open = QAction("Open Folder…", self)
        act_open.setShortcut("Ctrl+O")
        file_menu.addAction(act_open)
        file_menu.addSeparator()
        act_reload = QAction("Reload", self)
        act_reload.setShortcut("Ctrl+R")
        file_menu.addAction(act_reload)
        file_menu.addSeparator()
        act_exit = QAction("Exit", self)
        file_menu.addAction(act_exit)
        view_menu = menubar.addMenu("&View")
        act_toggle_side = QAction("Toggle Side Panel", self)
        view_menu.addAction(act_toggle_side)
        help_menu = menubar.addMenu("&Help")
        act_about = QAction("About", self)
        help_menu.addAction(act_about)
        # Toolbar
        toolbar = QToolBar("Main")
        toolbar.setIconSize(QSize(18, 18))
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        toolbar.addAction(act_open)
        toolbar.addAction(act_reload)
        act_plot = QAction("Plot", self)
        act_plot.setShortcut("Ctrl+P")
        act_export_sum = QAction("Export Summary", self)
        act_export_sum.setShortcut("Ctrl+S")
        toolbar.addAction(act_plot)
        toolbar.addAction(act_export_sum)
        # Shortcuts
        act_open.triggered.connect(self.open_folder)
        act_reload.triggered.connect(self.reload_folder)
        act_exit.triggered.connect(self.close)
        act_about.triggered.connect(self.show_about_dialog)
        act_plot.triggered.connect(self.plot_data)
        act_export_sum.triggered.connect(self.export_summary)

    def show_about_dialog(self):
        QMessageBox.about(self, "About Engine Test Data Explorer",
            "<b>Engine Test Data Explorer</b><br>"
            "PyQt5 desktop app for visualizing and summarizing engine test time-series data.<br><br>"
            "Copyright © 2025"
        )

    # Persistent settings
    def closeEvent(self, event):
        self.save_window_settings()
        super().closeEvent(event)
    def save_window_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("splitter", self.file_splitter.saveState())
        self.settings.setValue("last_folder", self.folder_path_edit.text())
    def restore_window_settings(self):
        if self.settings.contains("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))
        if self.settings.contains("windowState"):
            self.restoreState(self.settings.value("windowState"))
        if self.settings.contains("splitter"):
            self.file_splitter.restoreState(self.settings.value("splitter"))
    def restore_last_used_folder(self):
        last = self.settings.value("last_folder", "")
        if last and os.path.isdir(last):
            self.folder_path_edit.setText(last)
            self.scan_folder(last)

# [CONTINUE TO PART 2/3: FILE LOADING, FILTERING, AXIS TABLE, DYNAMIC CONTROLS, AND DATA LOADING...]

    # ============ File/folder logic ============
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder", self.folder_path_edit.text() or "")
        if not folder:
            return
        self.folder_path_edit.setText(folder)
        self.scan_folder(folder)

    def reload_folder(self):
        folder = self.folder_path_edit.text()
        if folder:
            self.scan_folder(folder)

    def scan_folder(self, folder):
        self.file_list.clear()
        self.file_list_data = []
        self.status_bar.showMessage("Scanning folder…")
        self.scanning = True
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.progress_bar.setRange(0, 0)
        self.scan_thread = FolderScanThread(folder)
        self.scan_thread.finished.connect(self.folder_scan_complete)
        self.scan_thread.start()
        QTimer.singleShot(60000, lambda: self.abort_scan())

    def abort_scan(self):
        if self.scanning:
            self.scan_thread.terminate()
            self.status_bar.showMessage("Scan aborted.")
            self.progress_bar.hide()
            self.scanning = False

    def folder_scan_complete(self, files):
        self.status_bar.clearMessage()
        self.progress_bar.hide()
        self.file_list_data = files
        self.scanning = False
        self.filter_files()
        self.status_bar.showMessage(f"Found {len(files)} data files.")
        self.save_window_settings()
        self.populate_column_choices()

    def filter_files(self):
        filter_txt = self.filter_edit.text().lower()
        self.file_list.clear()
        self.filtered_files = []
        for filepath in self.file_list_data:
            fname = os.path.basename(filepath)
            if filter_txt in fname.lower():
                item = QListWidgetItem(fname)
                item.setCheckState(Qt.Unchecked)
                item.setToolTip(filepath)
                self.file_list.addItem(item)
                self.filtered_files.append(filepath)

    def select_all_files(self):
        for i in range(self.file_list.count()):
            self.file_list.item(i).setCheckState(Qt.Checked)
    def deselect_all_files(self):
        for i in range(self.file_list.count()):
            self.file_list.item(i).setCheckState(Qt.Unchecked)
    def get_selected_files(self):
        selected = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(self.filtered_files[i])
        return selected
    def on_file_checked(self, item):
        self.populate_column_choices()

    # ============ Header preview ============
    def preview_headers(self):
        files = self.get_selected_files()
        if not files:
            QMessageBox.information(self, "Preview Headers", "No files selected.")
            return
        dlg = HeaderPreviewDialog(files, self)
        dlg.exec_()

    # ============ Dynamic Axis Table ============
    def init_dynamic_axis_table(self):
        # Always at least one row (primary Y)
        self.axis_table.setRowCount(1)
        self.axis_table.setItem(0, 0, QTableWidgetItem("Primary Y"))
        self.axis_table.setCellWidget(0, 1, QComboBox())
        remove_btn = QPushButton("✖")
        remove_btn.clicked.connect(lambda: self.remove_axis_row(0))
        self.axis_table.setCellWidget(0, 2, remove_btn)
        self.axis_table.setVerticalHeaderLabels(["Primary Y"])
        self.axis_rows = ["Primary Y"]
        self.axis_map = [("", "")]
        self.populate_column_choices()

    def add_axis_row(self):
        n = self.axis_table.rowCount()
        if n >= 4:
            QMessageBox.warning(self, "Axis Limit", "Maximum 4 Y axes supported.")
            return
        axis_names = ["Primary Y", "Secondary Y", "Tertiary Y", "Quaternary Y"]
        self.axis_table.insertRow(n)
        self.axis_table.setItem(n, 0, QTableWidgetItem(axis_names[n]))
        cb = QComboBox()
        self.axis_table.setCellWidget(n, 1, cb)
        remove_btn = QPushButton("✖")
        remove_btn.clicked.connect(lambda: self.remove_axis_row(n))
        self.axis_table.setCellWidget(n, 2, remove_btn)
        self.axis_table.setVerticalHeaderLabels(axis_names[:self.axis_table.rowCount()])
        self.axis_rows.append(axis_names[n])
        self.axis_map.append(("", ""))
        self.populate_column_choices()

    def remove_axis_row(self, idx):
        if self.axis_table.rowCount() <= 1:
            QMessageBox.warning(self, "Required", "At least one axis must be present.")
            return
        self.axis_table.removeRow(idx)
        self.axis_rows.pop(idx)
        self.axis_map.pop(idx)
        # Re-assign remove buttons and labels
        axis_names = ["Primary Y", "Secondary Y", "Tertiary Y", "Quaternary Y"]
        for i in range(self.axis_table.rowCount()):
            btn = QPushButton("✖")
            btn.clicked.connect(lambda _, x=i: self.remove_axis_row(x))
            self.axis_table.setCellWidget(i, 2, btn)
        self.axis_table.setVerticalHeaderLabels(axis_names[:self.axis_table.rowCount()])
        self.populate_column_choices()

    def axis_mode_changed(self):
        # Overlay: show only one plot; Separate: one subplot per axis
        pass  # handled in plot_data

    def axis_table_cell_changed(self, row, col):
        # For future: Save combo selection changes if needed
        pass

    def populate_column_choices(self):
        selected_files = self.get_selected_files()
        if not selected_files:
            all_columns = []
        else:
            all_columns = unique_numeric_columns(selected_files)
        if not all_columns:
            all_columns = []
        # For time/date combobox
        dt_cols = set()
        decimal_sep = self.decimal_sep_combo.currentText()
        for f in selected_files:
            try:
                df = pd.read_csv(f, sep='\t', nrows=5, decimal=decimal_sep)
                for c in df.columns:
                    if is_time_col(c):
                        dt_cols.add(c)
            except Exception:
                continue
        # Default fallback
        if not dt_cols and all_columns:
            dt_cols.add(all_columns[0])
        self.combo_datetime.clear()
        self.combo_datetime.addItems(sorted(dt_cols))
        # Axis combos
        for i in range(self.axis_table.rowCount()):
            cb = self.axis_table.cellWidget(i, 1)
            sel = cb.currentText() if cb.count() else ""
            cb.clear()
            cb.addItems(all_columns)
            if sel and sel in all_columns:
                cb.setCurrentText(sel)
            elif all_columns:
                cb.setCurrentIndex(i if i < len(all_columns) else 0)

    # ============ DataFrame loading ============
    def load_selected_dataframes(self):
        """Loads all selected files and keeps them in self.loaded_dfs, with exception handling."""
        files = self.get_selected_files()
        loaded = {}
        errors = []
        decimal_sep = self.decimal_sep_combo.currentText()
        for f in files:
            try:
                df = pd.read_csv(filepath, sep='\t', decimal=decimal_sep)
                loaded[f] = df
            except Exception as e:
                errors.append(f"{os.path.basename(f)}: {e}")
        self.loaded_dfs = loaded
        if errors:
            QMessageBox.warning(self, "File Load Errors", "\n".join(errors))

    # ============ Utility: Get axis/column mapping ============
    def get_axis_column_mapping(self):
        mapping = []
        for i in range(self.axis_table.rowCount()):
            cb = self.axis_table.cellWidget(i, 1)
            col = cb.currentText() if cb.count() else ""
            axis_name = self.axis_table.item(i, 0).text() if self.axis_table.item(i, 0) else f"Axis {i+1}"
            mapping.append((axis_name, col))
        return mapping

    # ============ Utility: Get selected time column ============
    def get_selected_time_column(self):
        return self.combo_datetime.currentText()

    # ============ CSV/Excel Export Dialogs ============
    def get_save_filepath(self, caption, exts):
        dlg = QFileDialog(self, caption, "", exts)
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            return dlg.selectedFiles()[0]
        return None

    # ============ Plotting ============
    def plot_data(self):
        self.load_selected_dataframes()
        files = self.get_selected_files()
        if not files:
            QMessageBox.information(self, "Plot", "No files selected.")
            return
        time_col = self.get_selected_time_column()
        axis_map = self.get_axis_column_mapping()
        if not time_col or not any(c for _, c in axis_map):
            QMessageBox.warning(self, "Missing Columns", "You must select a Date/Time column and at least one data column.")
            return

        overlay = self.radio_overlay.isChecked()
        autoscale = self.chk_autoscale.isChecked()
        self.figure.clear()
        if overlay:
            ax_main = self.figure.add_subplot(111)
            axes = [ax_main]
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
            twin_axes = [ax_main]
            for idx, (axis_name, col) in enumerate(axis_map):
                if not col: continue
                if idx == 0:
                    ax = ax_main
                else:
                    ax = ax_main.twinx()
                    # To avoid overlapping, offset right
                    ax.spines["right"].set_position(("axes", 1 + 0.07 * (idx - 1)))
                twin_axes.append(ax)
                color = colors[idx % len(colors)]
                label = f"{col}"
                for f in files:
                    df = self.loaded_dfs.get(f)
                    if df is None or col not in df.columns or time_col not in df.columns:
                        continue
                    tsec = extract_datetime_col(df, time_col)
                    y = pd.to_numeric(df[col], errors="coerce")
                    ax.plot(tsec, y, label=f"{os.path.basename(f)}:{col}", color=color)
                ax.set_ylabel(col)
                if autoscale: ax.autoscale()
            ax_main.set_xlabel("Time [s]")
            ax_main.legend(loc="upper right")
            self.figure.tight_layout()
        else:
            n_axes = sum(1 for _, col in axis_map if col)
            if n_axes == 0:
                QMessageBox.warning(self, "No Data", "No columns selected.")
                return
            axes = []
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
            for idx, (axis_name, col) in enumerate(axis_map):
                if not col: continue
                ax = self.figure.add_subplot(n_axes, 1, len(axes)+1)
                axes.append(ax)
                color = colors[idx % len(colors)]
                for f in files:
                    df = self.loaded_dfs.get(f)
                    if df is None or col not in df.columns or time_col not in df.columns:
                        continue
                    tsec = extract_datetime_col(df, time_col)
                    y = pd.to_numeric(df[col], errors="coerce")
                    ax.plot(tsec, y, label=f"{os.path.basename(f)}:{col}", color=color)
                ax.set_ylabel(col)
                if autoscale: ax.autoscale()
                ax.legend(loc="upper right")
            axes[-1].set_xlabel("Time [s]")
            self.figure.tight_layout()
        self.canvas.draw()
        self.status_bar.showMessage("Plot complete.", 3000)

    # ============ Right-click context menu ============
    def show_plot_context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("Save as PNG", self.export_plot_png)
        menu.addAction("Export Plot Data to CSV", self.export_plot_csv)
        menu.addAction("Export Current Summary", self.export_summary)
        menu.exec_(QCursor.pos())

    def export_plot_png(self):
        fname = self.get_save_filepath("Export Plot as PNG", "PNG Files (*.png);;All Files (*)")
        if not fname: return
        self.figure.savefig(fname)
        self.status_bar.showMessage(f"Plot exported: {fname}", 4000)

    def export_plot_csv(self):
        self.load_selected_dataframes()
        files = self.get_selected_files()
        if not files:
            QMessageBox.information(self, "Export", "No files selected.")
            return
        time_col = self.get_selected_time_column()
        axis_map = self.get_axis_column_mapping()
        out = []
        for f in files:
            df = self.loaded_dfs.get(f)
            if df is None or time_col not in df.columns: continue
            outdf = pd.DataFrame()
            outdf["Time [s]"] = extract_datetime_col(df, time_col)
            for _, col in axis_map:
                if col and col in df.columns:
                    outdf[col] = pd.to_numeric(df[col], errors="coerce")
            out.append(outdf)
        if not out:
            QMessageBox.warning(self, "Export", "No data to export.")
            return
        merged = pd.concat(out, axis=0, ignore_index=True)
        fname = self.get_save_filepath("Export Plot Data as CSV", "CSV Files (*.csv);;All Files (*)")
        if fname:
            merged.to_csv(fname, index=False)
            self.status_bar.showMessage(f"Data exported: {fname}", 4000)

    # ============ Summary Statistics ============
    def preview_summary(self):
        self.load_selected_dataframes()
        files = self.get_selected_files()
        if not files:
            QMessageBox.information(self, "Summary", "No files selected.")
            return
        time_col = self.get_selected_time_column()
        axis_map = self.get_axis_column_mapping()
        metric_types = self.get_selected_summary_metrics()
        start, end = self.get_time_window()
        summary_rows = []
        columns = []
        for f in files:
            df = self.loaded_dfs.get(f)
            if df is None or time_col not in df.columns: continue
            tsec = extract_datetime_col(df, time_col)
            mask = pd.Series([True]*len(df))
            if start is not None:
                mask &= (tsec >= start)
            if end is not None:
                mask &= (tsec <= end)
            row = [os.path.basename(f)]
            for _, col in axis_map:
                vals = pd.to_numeric(df[col][mask], errors="coerce") if col in df.columns else pd.Series([])
                for mt in metric_types:
                    if mt == "mean":
                        row.append(vals.mean() if not vals.empty else np.nan)
                    elif mt == "median":
                        row.append(vals.median() if not vals.empty else np.nan)
                    elif mt == "min":
                        row.append(vals.min() if not vals.empty else np.nan)
                    elif mt == "max":
                        row.append(vals.max() if not vals.empty else np.nan)
                    elif mt == "std":
                        row.append(vals.std() if not vals.empty else np.nan)
            summary_rows.append(row)
        # Compose columns
        columns = ["File"]
        for _, col in axis_map:
            for mt in metric_types:
                columns.append(f"{col}-{mt}")
        # Show summary in modal table
        dlg = QDialog(self)
        dlg.setWindowTitle("Summary Preview")
        lay = QVBoxLayout(dlg)
        table = QTableWidget(len(summary_rows), len(columns))
        table.setHorizontalHeaderLabels(columns)
        for i, row in enumerate(summary_rows):
            for j, val in enumerate(row):
                item = QTableWidgetItem("" if pd.isnull(val) else str(np.round(val, 6)))
                table.setItem(i, j, item)
        lay.addWidget(table)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.exec_()

    def get_selected_summary_metrics(self):
        v = self.combo_summary_type.currentText().lower()
        if v == "mean": return ["mean"]
        if v == "median": return ["median"]
        if v == "min/max": return ["min", "max"]
        if v == "std dev": return ["std"]
        if v.startswith("custom"):
            metrics, ok = QInputDialog.getText(self, "Custom Metrics",
                "Enter metrics (comma-separated, choices: mean, median, min, max, std):",
                text=",".join(self.custom_metrics_selected))
            if ok:
                chosen = [m.strip().lower() for m in metrics.split(",") if m.strip().lower() in self.custom_metrics]
                if chosen:
                    self.custom_metrics_selected = chosen
                    return chosen
            return ["mean"]
        return ["mean"]

    def get_time_window(self):
        def parse_time(s):
            s = s.strip()
            if not s: return None
            try:
                return float(s)
            except Exception:
                try:
                    return pd.to_datetime(s)
                except Exception:
                    return None
        s = parse_time(self.start_time_edit.text())
        e = parse_time(self.end_time_edit.text())
        # If both float: use as seconds, else None
        return (s if isinstance(s, float) else None, e if isinstance(e, float) else None)

    def export_summary(self):
        self.load_selected_dataframes()
        files = self.get_selected_files()
        if not files:
            QMessageBox.information(self, "Export", "No files selected.")
            return
        time_col = self.get_selected_time_column()
        axis_map = self.get_axis_column_mapping()
        metric_types = self.get_selected_summary_metrics()
        start, end = self.get_time_window()
        summary_rows = []
        columns = []
        for f in files:
            df = self.loaded_dfs.get(f)
            if df is None or time_col not in df.columns: continue
            tsec = extract_datetime_col(df, time_col)
            mask = pd.Series([True]*len(df))
            if start is not None:
                mask &= (tsec >= start)
            if end is not None:
                mask &= (tsec <= end)
            row = [os.path.basename(f)]
            for _, col in axis_map:
                vals = pd.to_numeric(df[col][mask], errors="coerce") if col in df.columns else pd.Series([])
                for mt in metric_types:
                    if mt == "mean":
                        row.append(vals.mean() if not vals.empty else np.nan)
                    elif mt == "median":
                        row.append(vals.median() if not vals.empty else np.nan)
                    elif mt == "min":
                        row.append(vals.min() if not vals.empty else np.nan)
                    elif mt == "max":
                        row.append(vals.max() if not vals.empty else np.nan)
                    elif mt == "std":
                        row.append(vals.std() if not vals.empty else np.nan)
            summary_rows.append(row)
        columns = ["File"]
        for _, col in axis_map:
            for mt in metric_types:
                columns.append(f"{col}-{mt}")
        fname = self.get_save_filepath("Export Summary Table", "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)")
        if not fname: return
        df_out = pd.DataFrame(summary_rows, columns=columns)
        try:
            if fname.lower().endswith(".xlsx"):
                df_out.to_excel(fname, index=False)
            else:
                df_out.to_csv(fname, index=False)
            self.status_bar.showMessage(f"Summary exported: {fname}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def summary_type_changed(self, idx):
        # If custom, prompt user
        if self.combo_summary_type.currentText().lower().startswith("custom"):
            self.get_selected_summary_metrics()

    # ============ MAIN ============
def main():
    import warnings
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = EngineTestDataExplorer()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
