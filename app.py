import sys
import json
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QMainWindow
from ui.ui_main_window import Ui_MainWindow
from ui.ui_review_window import Ui_reviewwindow
from Ml_pipeline import*

commentlist=[]
class WorkerThread(QThread):
    finished=Signal(list)
    error=Signal(str)
    progress=Signal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path=file_path
    def run(self):
        try:
            self.progress.emit("Analyzing code structure...please wait⏳")
            comments=comment_code(self.file_path)
            self.finished.emit(comments)
        except Exception as e:
            self.error.emit(f"Analysis failed:{str(e)}")



class MainWindow(QMainWindow):#subclass QMainWindow so it behaves like a widget not class
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()   #creates an instance based on the ui class
        self.ui.setupUi(self)   #setsup the interface on this window
        self.file=""
        self.ui.button.clicked.connect(self.on_submit)
        self.commentlist=[]
        self.reviewwindow=None
        self.worker_thread =None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():#mimeData==file that was dragged
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.file = urls[0].toLocalFile()
            if self.file.endswith(".py"):
                self.ui.dragdroplabel.setText(f"File dropped:\n{self.file}")
                print("File dropped:", self.file)
            else:
                self.ui.dragdroplabel.setText("Only .py files are supported.")

    def on_submit(self):
        if not self.file:
            self.ui.dragdroplabel.setText("Please select a .py file first.")
            return


        self.ui.button.setEnabled(False)

        QApplication.processEvents()

        self.worker_thread = WorkerThread(self.file)
        self.worker_thread.finished.connect(self.on_finished)
        self.worker_thread.error.connect(self.on_error)
        self.worker_thread.progress.connect(self.on_progress)
        self.worker_thread.start()

    def open_reviewwindow(self):
        if self.reviewwindow is None:
            self.reviewwindow = ReviewWindow(self.commentlist,self.file)
            self.reviewwindow.show()
        else:
            self.reviewwindow.update_comments(self.commentlist)
            self.reviewwindow.show()
            self.reviewwindow.raise_()

    def on_finished(self, comments):
        self.commentlist=comments
        self.ui.button.setEnabled(True)
        self.ui.dragdroplabel.setText(f"analysis complete found {len(comments)} classes and functions")
        self.open_reviewwindow()

    def on_error(self,error):
        self.ui.button.setEnabled(True)
        self.ui.dragdroplabel.setText(f"analysis failed, Error: {error}")

    def on_progress(self,progress):
        self.ui.dragdroplabel.setText(progress)


class ReviewWindow(QMainWindow):
    def __init__(self,commentlist,file_path, parent=None):
        super().__init__(parent)
        self.ui = Ui_reviewwindow()
        self.ui.setupUi(self)
        self.file_path=file_path
        self.text_edits,self.originalcomments,self.nodenames=reviewcomments(self.ui.scrollArea,commentlist)
        self.ui.pushButton.clicked.connect(self.on_submit)

    def update_comments(self,newcommentlist):#update comments if submit is pressed multiple times
        for text in self.text_edits:
            text.deleteLater()#deletes old comment widgets
        self.text_edits=reviewcomments(self.ui.scrollArea,newcommentlist)#creates new widgets
        self.originalcomments = [commentlist[i][1] for i in range(len(commentlist))]

    def save_training_data(self, original_comments, edited_comments):
        training_data = []
        for orig, edited in zip(original_comments, edited_comments):
            # Only save if user actually made changes
            if orig != edited:
                training_data.append({
                    "original_comment": orig,
                    "edited_comment": edited
                })
        # Append to file
        with open("training_data.jsonl", "a") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")

    def write_to_file(self,file_path, comments_dict):
        with open(file_path, 'r', encoding='utf-8') as f:
           content = f.read()

        tree = ast.parse(content)

        # Modify the AST
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name in comments_dict:
                    # Create docstring node
                    docstring_node = ast.Expr(value=ast.Constant(value=comments_dict[node.name]))
                    # Insert docstring as first element in body
                    node.body.insert(0, docstring_node)

        # Convert AST back to code
        new_content = ast.unparse(tree)
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    def on_submit(self):
        edited_comments = [te.toPlainText() for te in self.text_edits]
        commentdict = dict(zip(self.nodenames, edited_comments))
        self.save_training_data(self.originalcomments,edited_comments)
        self.write_to_file(self.file_path, commentdict)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())



