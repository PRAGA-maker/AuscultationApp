import sys
import os
import time
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QRadioButton,
                             QButtonGroup, QComboBox, QTabWidget, QSizePolicy, QGroupBox, QMessageBox, QSlider)
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from stockwell import st
import matplotlib.pyplot as plt

class AnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.current_index = -1  # Track the current file index
        self.annotations = []
        self.start_time = None
        self.exit_flag = False
        self.line_positions = {'S1_start': None, 'S1_end': None, 'S2_start': None, 'S2_end': None}
        self.quality_drop_positions = []  # List to store multiple quality drop pairs
        self.lines = []
        self.line_labels = []
        self.marking_type = None  # To handle marking type selection
        self.s_transform_used = False  # To track if S-transform was used
        self.file_list = []  # List of files to be annotated
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.LowLatency) 
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Annotate Spectrogram')
        self.setStyleSheet("""
            QMainWindow {
                background-color: #e9e4d4;
                color: #3b3b4f;
            }
            QLabel, QRadioButton, QPushButton, QComboBox {
                color: #3b3b4f;
                font-family: 'Arial';
                font-size: 14px;
            }
            QPushButton {
                background-color: #a3b18a;
                border: none;
                padding: 10px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #d6d2c4;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                padding: 5px;
            }
            QTabBar::tab {
                background: #a3b18a;
                padding: 10px;
            }
            QTabBar::tab:selected {
                background: #d6d2c4;
            }
        """)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        layout = QVBoxLayout(self.main_widget)

        self.canvas = FigureCanvas(Figure())
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.ax = self.canvas.figure.subplots()
        self.canvas.mpl_connect('button_press_event', self.on_click)

        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.create_annotation_tab()
        self.create_controls_tab()

    def create_annotation_tab(self):
        annotation_tab = QWidget()
        annotation_layout = QVBoxLayout(annotation_tab)

        self.view_type = QComboBox()
        self.view_type.addItems(["Spectrogram", "S-Transform", "Dual View"])
        self.view_type.currentTextChanged.connect(self.confirm_update_view)
        annotation_layout.addWidget(QLabel("View Type:"))
        annotation_layout.addWidget(self.view_type)

        self.quality_group = QButtonGroup(self)
        self.quality_unsure = QRadioButton("unsure")
        self.quality_good = QRadioButton("Good")
        self.quality_bad = QRadioButton("Bad")
        self.quality_group.addButton(self.quality_good)
        self.quality_group.addButton(self.quality_bad)
        self.quality_group.addButton(self.quality_unsure)

        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        quality_layout.addWidget(self.quality_good)
        quality_layout.addWidget(self.quality_bad)
        quality_layout.addWidget(self.quality_unsure)
        annotation_layout.addLayout(quality_layout)

        self.systolic_murmur_group = QButtonGroup(self)
        self.systolic_murmur_yes = QRadioButton("yes")
        self.systolic_murmur_no = QRadioButton("no")
        self.systolic_murmur_unsure = QRadioButton("unsure")
        self.systolic_murmur_group.addButton(self.systolic_murmur_yes)
        self.systolic_murmur_group.addButton(self.systolic_murmur_no)
        self.systolic_murmur_group.addButton(self.systolic_murmur_unsure)

        systolic_layout = QHBoxLayout()
        systolic_layout.addWidget(QLabel("Systolic Murmur:"))
        systolic_layout.addWidget(self.systolic_murmur_yes)
        systolic_layout.addWidget(self.systolic_murmur_no)
        systolic_layout.addWidget(self.systolic_murmur_unsure)
        annotation_layout.addLayout(systolic_layout)

        self.diastolic_murmur_group = QButtonGroup(self)
        self.diastolic_murmur_yes = QRadioButton("yes")
        self.diastolic_murmur_no = QRadioButton("no")
        self.diastolic_murmur_unsure = QRadioButton("unsure")
        self.diastolic_murmur_group.addButton(self.diastolic_murmur_yes)
        self.diastolic_murmur_group.addButton(self.diastolic_murmur_no)
        self.diastolic_murmur_group.addButton(self.diastolic_murmur_unsure)

        diastolic_layout = QHBoxLayout()
        diastolic_layout.addWidget(QLabel("Diastolic Murmur:"))
        diastolic_layout.addWidget(self.diastolic_murmur_yes)
        diastolic_layout.addWidget(self.diastolic_murmur_no)
        diastolic_layout.addWidget(self.diastolic_murmur_unsure)
        annotation_layout.addLayout(diastolic_layout)

        self.continuous_murmur_group = QButtonGroup(self)
        self.continuous_murmur_yes = QRadioButton("yes")
        self.continuous_murmur_no = QRadioButton("no")
        self.continuous_murmur_unsure = QRadioButton("unsure")
        self.continuous_murmur_group.addButton(self.continuous_murmur_yes)
        self.continuous_murmur_group.addButton(self.continuous_murmur_no)
        self.continuous_murmur_group.addButton(self.continuous_murmur_unsure)

        continuous_layout = QHBoxLayout()
        continuous_layout.addWidget(QLabel("Continuous Murmur:"))
        continuous_layout.addWidget(self.continuous_murmur_yes)
        continuous_layout.addWidget(self.continuous_murmur_no)
        continuous_layout.addWidget(self.continuous_murmur_unsure)
        annotation_layout.addLayout(continuous_layout)

        self.quality_unsure.setChecked(True)
        self.systolic_murmur_unsure.setChecked(True)
        self.diastolic_murmur_unsure.setChecked(True)
        self.continuous_murmur_unsure.setChecked(True)

        annotation_layout.addWidget(QLabel("Annotation Confidence:"))
        self.confidence_dropdown = QComboBox()
        self.confidence_dropdown.addItems(["Perfect", "High", "Low", "None"])
        annotation_layout.addWidget(self.confidence_dropdown)

        self.drop_quality_group = QButtonGroup(self)
        self.drop_quality_temporary = QRadioButton("Temporary")
        self.drop_quality_permanent = QRadioButton("Permanent")
        self.drop_quality_none = QRadioButton("None")
        self.drop_quality_group.addButton(self.drop_quality_temporary)
        self.drop_quality_group.addButton(self.drop_quality_permanent)
        self.drop_quality_group.addButton(self.drop_quality_none)
        
        drop_quality_layout = QHBoxLayout()
        drop_quality_layout.addWidget(QLabel("Drop in Quality:"))
        drop_quality_layout.addWidget(self.drop_quality_temporary)
        drop_quality_layout.addWidget(self.drop_quality_permanent)
        drop_quality_layout.addWidget(self.drop_quality_none)
        annotation_layout.addLayout(drop_quality_layout)
        
        self.drop_quality_none.setChecked(True)

        self.marking_type_group = QButtonGroup(self)
        self.marking_type_group.setExclusive(False)  # Allow no selection
        self.mark_timings = QRadioButton("Mark Timings")
        self.mark_quality = QRadioButton("Mark Quality Drops")
        self.marking_type_group.addButton(self.mark_timings)
        self.marking_type_group.addButton(self.mark_quality)

        marking_type_box = QGroupBox("Marking Type")
        marking_type_box.setStyleSheet("QGroupBox { border: 2px solid darkgreen; border-radius: 5px; margin-top: 10px; }")
        marking_type_layout = QVBoxLayout()
        marking_type_layout.addWidget(self.mark_timings)
        marking_type_layout.addWidget(self.mark_quality)
        marking_type_box.setLayout(marking_type_layout)
        annotation_layout.addWidget(marking_type_box)

        self.tabs.addTab(annotation_tab, "Annotations")

    def create_controls_tab(self):
        controls_tab = QWidget()
        controls_layout = QVBoxLayout(controls_tab)

        buttons_layout = QHBoxLayout()
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(lambda: self.save_annotations(skip=False))
        buttons_layout.addWidget(self.next_button)

        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.go_back)
        buttons_layout.addWidget(self.back_button)

        self.skip_button = QPushButton("Skip")
        self.skip_button.clicked.connect(lambda: self.save_annotations(skip=True))
        buttons_layout.addWidget(self.skip_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(lambda: self.save_annotations(skip=False, exit=True))
        buttons_layout.addWidget(self.exit_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_annotations)
        buttons_layout.addWidget(self.reset_button)

        controls_layout.addLayout(buttons_layout)

        audio_layout = QHBoxLayout()
        self.playButton = QPushButton('▶')
        self.playButton.setFixedWidth(30)
        self.playButton.clicked.connect(self.toggle_play)
        audio_layout.addWidget(self.playButton)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.sliderMoved.connect(self.set_position)
        audio_layout.addWidget(self.slider)

        self.timeLabel = QLabel('0:00 / 0:00')
        self.timeLabel.setFixedWidth(100)
        audio_layout.addWidget(self.timeLabel)

        controls_layout.addLayout(audio_layout)

        self.mediaPlayer.durationChanged.connect(self.update_duration)
        self.mediaPlayer.positionChanged.connect(self.update_position)

        self.timer = QTimer(self)
        self.timer.setInterval(5)  
        self.timer.timeout.connect(lambda: self.update_audio_line(self.mediaPlayer.position()))
        self.timer.timeout.connect(self.update_slider)
        self.timer.start()

        self.tabs.addTab(controls_tab, "Controls")

    def confirm_update_view(self):
        view = self.view_type.currentText()
        if view == "S-Transform":
            reply = QMessageBox.question(self, 'Confirmation',
                                         "Are you sure you want to switch to S-Transform view?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:    
                self.s_transform_used = True
                self.update_view()
            else:
                self.view_type.blockSignals(True)
                self.view_type.setCurrentIndex(0)  # Revert to Spectrogram view if 'No' 
                self.view_type.blockSignals(False)
        else:
            self.update_view()

    def update_view(self):
        self.start_time = time.time()
        view = self.view_type.currentText()
        if view == "Spectrogram":
            self.show_spectrogram(self.ax, self.current_file)
        elif view == "S-Transform":
            self.show_s_transform(self.ax, self.current_file)
        elif view == "Dual View":
            self.show_dual_view(self.ax, self.current_file)
        self.canvas.draw()

    def show_spectrogram(self, ax, filepath):
        ax.clear()
        rate, data = wav.read(filepath)
        if data.ndim > 1:  
            data = np.mean(data, axis=1)
        Pxx, freqs, bins, im = ax.specgram(data, Fs=rate, NFFT=1024, noverlap=900, cmap='jet')
        Pxx[Pxx == 0] = np.finfo(float).eps  # Prevent log(0) issues
        ax.imshow(10 * np.log10(Pxx), extent=[0, bins[-1], freqs[0], freqs[-1]], aspect='auto', cmap='jet', origin='lower')
        ax.set_title(os.path.basename(filepath), pad=30)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        self.restore_lines(ax)

    def show_s_transform(self, ax, filepath, max_length=False, downsample_factor=10):
        ax.clear()
        rate, data = wav.read(filepath)
        if data.ndim > 1:  
            data = np.mean(data, axis=1)
        data = data[::downsample_factor]
        rate = rate // downsample_factor
        if max_length:
            data = data[:rate * max_length]
        S = st.st(data)
        ax.imshow(np.abs(S), aspect='auto', extent=[0, len(data)/rate, 0, rate/2], cmap='jet', origin='lower')
        ax.set_title(f'S-Transform: {os.path.basename(filepath)}', pad=30)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_yscale('log')
        ax.set_ylim([10, rate / 2])
        self.restore_lines(ax)

    def show_dual_view(self, ax, filepath):
        ax.clear()
        rate, data = wav.read(filepath)
        time = np.linspace(0, len(data) / rate, num=len(data))
        ax.plot(time, data)
        ax.set_title(f'Dual View: {os.path.basename(filepath)}', pad=30)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        self.restore_lines(ax)

    def restore_lines(self, ax):
        self.lines = []
        self.line_labels = []
        for key, time in self.line_positions.items():
            if time is not None:
                if 'S1_start' in key:
                    line = ax.axvline(time, color='red', label='S1 Start') 
                    label = 'S1 Start'
                elif 'S1_end' in key:
                    line = ax.axvline(time, color='blue', label='S1 End')
                    label = 'S1 End'
                elif 'S2_start' in key:
                    line = ax.axvline(time, color='green', label='S2 Start')
                    label = 'S2 Start'
                elif 'S2_end' in key:
                    line = ax.axvline(time, color='purple', label='S2 End')
                    label = 'S2 End'
                self.lines.append(line)
                line_label = self.ax.text(time, ax.get_ylim()[1], label, color=line.get_color(), verticalalignment='bottom')
                self.line_labels.append(line_label)
        for start, end in self.quality_drop_positions:
            if start is not None:
                line = ax.axvline(start, color='orange', linestyle='--', label='Quality Drop Start')
                label = 'Quality Drop Start'
                self.lines.append(line)
                line_label = self.ax.text(start, self.ax.get_ylim()[1], label, color=line.get_color(), verticalalignment='bottom')
                self.line_labels.append(line_label)
            if end is not None:
                line = ax.axvline(end, color='brown', linestyle='--', label='Quality Drop End')
                label = 'Quality Drop End'
                self.lines.append(line)
                line_label = self.ax.text(end, self.ax.get_ylim()[1], label, color=line.get_color(), verticalalignment='bottom')
                self.line_labels.append(line_label)

    def save_annotations(self, skip=False, exit=False):
        end_time = time.time()
        if self.start_time is None:
            self.start_time = end_time
        time_spent = end_time - self.start_time

        annotation = {
            'filename': self.current_file,
            'quality': self.quality_group.checkedButton().text() if self.quality_group.checkedButton() else 'skipped',
            'systolic_murmur': self.systolic_murmur_group.checkedButton().text() if self.systolic_murmur_group.checkedButton() else 'skipped',
            'diastolic_murmur': self.diastolic_murmur_group.checkedButton().text() if self.diastolic_murmur_group.checkedButton() else 'skipped',
            'continuous_murmur': self.continuous_murmur_group.checkedButton().text() if self.continuous_murmur_group.checkedButton() else 'skipped',
            'confidence': self.confidence_dropdown.currentText() if not skip else 'skipped',
            'quality_drop': self.drop_quality_group.checkedButton().text() if self.drop_quality_group.checkedButton() else 'skipped',
            'time_spent': time_spent,
            's_transform_used': self.s_transform_used  # 
        }
        annotation.update(self.line_positions)
        # Save       quality drop positions as a list of tuples
        annotation['quality_drop_positions'] = self.quality_drop_positions
        self.annotations.append(annotation)

        if not exit:
            self.reset_annotations()
            self.load_next_file()
        else:
            self.exit_flag = True
            self.close()

    def reset_annotations(self):
        self.quality_unsure.setChecked(True)
        self.systolic_murmur_unsure.setChecked(True)
        self.diastolic_murmur_unsure.setChecked(True)
        self.continuous_murmur_unsure.setChecked(True)
        self.line_positions = {'S1_start': None, 'S1_end': None, 'S2_start': None, 'S2_end': None}
        self.quality_drop_positions = []
        self.confidence_dropdown.setCurrentIndex(3)
        self.drop_quality_temporary.setChecked(True)
        self.marking_type_group.setExclusive(False)
        self.mark_timings.setChecked(False)
        self.mark_quality.setChecked(False)
        self.marking_type_group.setExclusive(True)
        self.s_transform_used = False  # Reset S-transform 
        for line in self.lines:
            line.remove()
        for label in self.line_labels:
            label.remove()
        self.lines.clear()
        self.line_labels.clear()
        self.canvas.draw()

    def load_next_file(self):
        if self.current_index < len(self.file_list) - 1:
            self.current_index += 1
            self.current_file = self.file_list[self.current_index]
            self.set_audio_file(self.current_file) 
            self.update_view()
        else:
            self.close()

    def load_previous_file(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.current_file = self.file_list[self.current_index]
            self.set_audio_file(self.current_file) 
            self.update_view()

    def get_completed_files(self, csv_path):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df['filename'].tolist()
        return []

    def get_file_list(self, folder_path, completed_files):
        self.file_list = []
        for subdir, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.wav'):
                    full_path = os.path.join(subdir, file)
                    if full_path not in completed_files:
                        self.file_list.append(full_path)
        return self.file_list

    def on_click(self, event):
        if event.inaxes != self.ax or not self.marking_type_group.checkedButton():
            return
        if self.toolbar.mode != '':
            # If any toolbar tools are active, reset the marking type
            self.marking_type_group.setExclusive(False)
            self.mark_timings.setChecked(False)
            self.mark_quality.setChecked(False)
            self.marking_type_group.setExclusive(True)
            return
        time = event.xdata
        label = ""
        if self.mark_timings.isChecked():
            if self.line_positions['S1_start'] is None:
                self.line_positions['S1_start'] = time
                line = self.ax.axvline(time, color='red', label='S1 Start')
                label = 'S1 Start'
            elif self.line_positions['S1_end'] is None:
                self.line_positions['S1_end'] = time
                line = self.ax.axvline(time, color='blue', label='S1 End')
                label = 'S1 End'
            elif self.line_positions['S2_start'] is None:
                self.line_positions['S2_start'] = time
                line = self.ax.axvline(time, color='green', label='S2 Start')
                label = 'S2 Start'
            elif self.line_positions['S2_end'] is None:
                self.line_positions['S2_end'] = time
                line = self.ax.axvline(time, color='purple', label='S2 End')
                label = 'S2 End'
        elif self.mark_quality.isChecked():
            if not self.quality_drop_positions or self.quality_drop_positions[-1][1] is not None:
                self.quality_drop_positions.append([time, None])
                line = self.ax.axvline(time, color='orange', linestyle='--', label='Quality Drop Start')
                label = 'Quality Drop Start'
            elif self.quality_drop_positions[-1][1] is None:
                self.quality_drop_positions[-1][1] = time
                line = self.ax.axvline(time, color='brown', linestyle='--', label='Quality Drop End')
                label = 'Quality Drop End'
        if label:
            self.lines.append(line)
            line_label = self.ax.text(time, self.ax.get_ylim()[1], label, color=line.get_color(), verticalalignment='bottom')
            self.line_labels.append(line_label)
        self.canvas.draw()

    def go_back(self):
        if self.current_index > 0:
            # Remove the last annotation
            self.annotations.pop()
            self.load_previous_file()
            self.reset_annotations()
            self.update_view()

    def set_audio_file(self, file_path):
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(file_path))))

    def toggle_play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.playButton.setText('▶')
        else:
            self.mediaPlayer.play()
            self.playButton.setText('⏸')

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def update_duration(self, duration):
        self.slider.setRange(0, duration)
        self.update_time_label(duration)

    def update_position(self, position):
        self.slider.setValue(position)
        self.update_time_label(position)
        self.update_audio_line(position)

    def update_slider(self):
        if not self.slider.isSliderDown():
            self.slider.setValue(self.mediaPlayer.position())

    def update_time_label(self, position):
        duration = self.mediaPlayer.duration()
        seconds = position // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        total_seconds = duration // 1000
        total_minutes = total_seconds // 60
        total_seconds = total_seconds % 60
        self.timeLabel.setText(f"{minutes}:{seconds:02d} / {total_minutes}:{total_seconds:02d}")

    def update_audio_line(self, position):
        # Clear any existing audio line
        for line in self.lines:
            if line.get_linestyle() == '-.':
                line.remove()
        self.lines = [line for line in self.lines if line.get_linestyle() != '-.']

        # Add new audio line
        time = position / 1000  # Convert position to seconds
        audio_line = self.ax.axvline(time, color='magenta', linestyle='-.', label='Audio Position')
        self.lines.append(audio_line)
        self.canvas.draw()


def annotate_spectrograms(folder_path, csv_path):
    app = QApplication(sys.argv)
    window = AnnotationApp()
    
    completed_files = window.get_completed_files(csv_path)  
    window.get_file_list(folder_path, completed_files)  
    
    if window.file_list:
        window.show()
        window.load_next_file()  # Load the first unannotated file
        app.exec_()

        df = pd.DataFrame(window.annotations)
        
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df = pd.concat([df_existing, df], ignore_index=True) 
        
        df.to_csv(csv_path, index=False)
    else:
        print("All files have been annotated.")

if __name__ == '__main__':
    folder_path = r"C:\Users\prapa\Desktop\other\auscultation\lamata\PraneelData"  # Change path
    csv_path = r"C:\Users\prapa\Desktop\other\auscultation\lamata\data.csv"  # Change path
    annotate_spectrograms(folder_path, csv_path)
