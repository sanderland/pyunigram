# --- model.py_integration_gui.py ---
import math
import sys

# --- PyQt6 Imports ---
from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# --- Import the Lattice and HumanTrainerModel classes ---
from py_unigram.human.model import Lattice, Token, UnigramModel

# --- GUI Logic ---


class LatticeNodeWidget(QGraphicsRectItem):
    def __init__(self, token: Token, token_pos: int, lattice: Lattice):
        super().__init__()
        """Visual node for a lattice token."""
        self.token = token
        self.token_pos = token_pos
        self.lattice = lattice
        self.setRect(0, 0, 120, 70)
        self.initial_brush = QBrush(QColor("#e0e0e0"))
        self.selected_brush = QBrush(QColor("#a0c4ff"))
        self.setBrush(self.initial_brush)
        self.setPen(QPen(Qt.GlobalColor.black, 1))
        self.text_item = QGraphicsTextItem(parent=self)
        self.text_item.setDefaultTextColor(Qt.GlobalColor.black)
        self.text_item.setPos(5, 5)
        self.update_display()

    def update_display(self):
        """Update token display text and size."""
        piece_str = self.token.text
        # Gamma: store on token if available, else 0
        gamma_val = getattr(self.token, 'gamma', 0.0)
        gamma_text = f"γ:{gamma_val:.3f}" if gamma_val > 1e-6 else "γ:0.000"
        self.text_item.setPlainText(
            f"'{piece_str}'\n"
            f"[{self.token_pos}-{self.token_pos + len(self.token.text)})\n"
            f"{gamma_text}"
        )
        text_rect = self.text_item.boundingRect()
        if text_rect.width() > self.rect().width() - 10:
            self.setRect(0, 0, text_rect.width() + 10, self.rect().height())
        if text_rect.height() > self.rect().height() - 10:
            self.setRect(0, 0, self.rect().width(), text_rect.height() + 10)

    def set_selected(self, is_selected):
        if is_selected:
            self.setBrush(self.selected_brush)
        else:
            self.setBrush(self.initial_brush)
        self.update()



class LatticeView(QGraphicsView):
    """Lattice visualization widget."""
    node_selected = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lattice = None
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.nodes_widgets = {}
        self.selected_node_widget = None
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

    def update_view(self, lattice: Lattice, _unused=None):
        self.lattice = lattice
        self.scene.clear()
        self.nodes_widgets = {}
        self.selected_node_widget = None
        if not self.lattice:
            return
        N = len(self.lattice.text)
        if N == 0:
            return
        max_y = 0
        # Place tokens as nodes
        for pos, tokens in enumerate(self.lattice.tokens_from_pos):
            for token in tokens:
                node_widget = LatticeNodeWidget(token, pos, self.lattice)
                self.nodes_widgets[(pos, token.id)] = node_widget
                self.scene.addItem(node_widget)
                x = pos * 130
                y = (pos + len(token.text)) * 80
                if y > max_y:
                    max_y = y
                node_widget.setPos(x, y)
        pen = QPen(QColor("#1976d2"), 1.5, Qt.PenStyle.SolidLine)
        for pos, tokens in enumerate(self.lattice.tokens_from_pos):
            for token in tokens:
                start_widget = self.nodes_widgets.get((pos, token.id))
                if not start_widget:
                    continue
                end_pos = pos + len(token.text)
                if end_pos < len(self.lattice.tokens_from_pos):
                    for next_token in self.lattice.tokens_from_pos[end_pos]:
                        end_widget = self.nodes_widgets.get((end_pos, next_token.id))
                        if not end_widget:
                            continue
                        start_point_scene = start_widget.mapToScene(start_widget.boundingRect().center().x(), start_widget.boundingRect().bottom())
                        end_point_scene = end_widget.mapToScene(end_widget.boundingRect().center().x(), end_widget.boundingRect().top())
                        arrow = QGraphicsLineItem(
                            start_point_scene.x(), start_point_scene.y(),
                            end_point_scene.x(), end_point_scene.y()
                        )
                        arrow.setPen(pen)
                        self.scene.addItem(arrow)
        scene_width = max(800, (N + 2) * 130)
        scene_height = max(600, max_y + 200)
        self.scene.setSceneRect(0, 0, scene_width, scene_height)

    def mousePressEvent(self, event):
        if self.selected_node_widget:
            self.selected_node_widget.set_selected(False)
            self.selected_node_widget = None
        clicked_item = self.itemAt(event.pos())
        token_to_emit = None
        if clicked_item:
            top_item = clicked_item
            while top_item and not isinstance(top_item, LatticeNodeWidget):
                top_item = top_item.parentItem() if hasattr(top_item, 'parentItem') else None
            if isinstance(top_item, LatticeNodeWidget):
                top_item.set_selected(True)
                self.selected_node_widget = top_item
                token_to_emit = (top_item.token, top_item.token_pos)
        self.node_selected.emit(token_to_emit)
        super().mousePressEvent(event)




class HumanUnigramFBModel:
    """Coordinator for the Forward-Backward process using human.model."""
    def __init__(self, vocab_scores: dict):
        self.tokens = [Token(text=k, id=i, log_prob=v) for i, (k, v) in enumerate(vocab_scores.items())]
        self.model = UnigramModel(self.tokens)
        self.sentence = ""
        self.lattice: Lattice = None
        self.token_probs = {}

    def run_full_algorithm(self, sentence: str):
        self.sentence = sentence
        if not self.sentence:
            return
        self.lattice = self.model.make_lattice(self.sentence)
        # Compute marginal probabilities for each token occurrence
        z, token_probs = self.lattice.calc_marginal() if self.lattice else {}
        # Attach gamma to each token occurrence in the lattice
        for pos, tokens in enumerate(self.lattice.tokens_from_pos):
            for token in tokens:
                gamma = token_probs.get(token.id, float('nan'))
                setattr(token, 'gamma', gamma)
        self.token_probs = token_probs



class ForwardBackwardVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Forward-Backward Visualizer (Human Unigram LM)")
        self.setGeometry(100, 100, 1600, 1000)
        self.fb_model = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Control Panel ---
        control_panel = QFrame()
        control_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Vocabulary Input
        vocab_group = QGroupBox("Vocabulary (Piece : Log Score)")
        vocab_layout = QVBoxLayout(vocab_group)
        self.vocab_text = QTextEdit()
        self.vocab_text.setMaximumHeight(150)
        self.vocab_text.setPlaceholderText("piece1 -1.0\npiece2 -2.5\nchar1 -0.5")
        # Default matching the user's example
        default_vocab = "h -0.5\ne -0.4\nl -0.6\no -0.5\nhe -1.2\nel -1.3\nll -1.4\nlo -1.1\nhel -2.0\nell -2.1\nllo -2.2\nhello -3.0"
        self.vocab_text.setText(default_vocab)
        vocab_layout.addWidget(self.vocab_text)

        # Sentence Input
        sentence_group = QGroupBox("Input Sentence")
        sentence_layout = QVBoxLayout(sentence_group)
        self.sentence_input = QTextEdit()
        self.sentence_input.setMaximumHeight(50)
        self.sentence_input.setText("hello") # Default sentence
        sentence_layout.addWidget(self.sentence_input)

        # Controls
        run_button = QPushButton("Run Forward-Backward")
        run_button.clicked.connect(self.run_algorithm)
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(run_button)

        # Results Display
        results_group = QGroupBox("Overall Results")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(180)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)

        # Node Details Display
        detail_group = QGroupBox("Node Details (Click a node)")
        detail_layout = QVBoxLayout(detail_group)
        self.detail_text = QTextEdit()
        self.detail_text.setMaximumHeight(300) # Increased height
        self.detail_text.setReadOnly(True)
        detail_layout.addWidget(self.detail_text)

        control_layout.addWidget(vocab_group)
        control_layout.addWidget(sentence_group)
        control_layout.addLayout(controls_layout)
        control_layout.addWidget(results_group)
        control_layout.addWidget(detail_group)
        control_layout.addStretch()

        # --- Right Visualization Panel ---
        viz_splitter = QSplitter(Qt.Orientation.Vertical)

        # Lattice View (Top part of right panel)
        self.lattice_view = LatticeView()
        self.lattice_view.node_selected.connect(self.on_node_selected)
        viz_splitter.addWidget(self.lattice_view)

        # Explanation Section (Bottom part of right panel)
        explanation_text = QTextEdit()
        explanation_text.setReadOnly(True)
        explanation_text.setMaximumHeight(200)
        explanation_text.setHtml(
            "<b>What are Alpha, Beta, and Gamma?</b><br>"
            "<i>Lattice</i>: A graph where nodes represent possible subword pieces at specific positions in the sentence. Arrows show valid transitions.<br><br>"
            "<i>Alpha (α)</i>: For a node, α = the log-probability of the <i>best path</i> that ends at the <i>start</i> of this node. "
            "It's calculated by the Viterbi algorithm during <code>populate_marginal</code>.<br><br>"
            "<i>Beta (β)</i>: For a node, β = the log-probability of the <i>best path</i> from the <i>end</i> of this node to the end of the sentence. "
            "It's also calculated during <code>populate_marginal</code>.<br><br>"
            "<i>Marginal Probability (γ)</i>: γ(node) = P(this specific node is used in the sentence) "
            "= exp(α(node_start) + logP(piece) + β(node_end) - logP(sentence)). "
            "This tells us how 'important' this specific piece occurrence is, considering <i>all possible</i> segmentations."
        )
        viz_splitter.addWidget(explanation_text)

        # Add parts to the right-side splitter
        viz_splitter.setSizes([700, 300])

        # Add panels to main layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(viz_splitter)
        splitter.setSizes([400, 1200])
        main_layout.addWidget(splitter)

    def parse_vocab(self):
        text = self.vocab_text.toPlainText()
        vocab = {}
        errors = []
        for line in text.strip().split('\n'):
            if line.strip():
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    piece, score_str = parts
                    piece = piece.strip()
                    try:
                        score = float(score_str.strip())
                        if piece in vocab:
                            errors.append(f"Duplicate piece '{piece}', keeping first.")
                        else:
                            vocab[piece] = score
                    except ValueError:
                        errors.append(f"Invalid score '{score_str}' for piece '{piece}'.")
                else:
                    errors.append(f"Invalid line format: '{line}'")
        if errors:
             QMessageBox.warning(self, "Vocabulary Parse Errors", "\n".join(errors))
        return vocab

    def run_algorithm(self):
        vocab_dict = self.parse_vocab()
        if not vocab_dict:
            QMessageBox.warning(self, "Error", "Vocabulary is empty or invalid.")
            return
        sentence = self.sentence_input.toPlainText().strip()
        if not sentence:
            QMessageBox.warning(self, "Error", "Please enter a sentence.")
            return
        try:
            self.fb_model = HumanUnigramFBModel(vocab_dict)
            self.fb_model.run_full_algorithm(sentence)
            self.lattice_view.update_view(self.fb_model.lattice)
            self.update_overall_results()
            self.detail_text.setPlainText("Click on a node in the lattice to see its details.")
        except Exception as e:
            QMessageBox.critical(self, "Algorithm Error", f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

    def update_overall_results(self):
        if not self.fb_model or not self.fb_model.lattice:
            self.results_text.setPlainText("")
            return
        result_text = f"Sentence: '{self.fb_model.sentence}'\n"
        # Show log probability if available
        try:
            alpha, _ = self.fb_model.lattice._forward_backward()
            log_Z = alpha[-1]
            if log_Z == float('-inf'):
                result_text += "Log Probability P(Sentence): -inf\n"
                result_text += "Probability P(Sentence): 0.0\n"
            else:
                result_text += f"Log Probability P(Sentence): {log_Z:.4f}\n"
                result_text += f"Probability P(Sentence): {math.exp(log_Z):.4e}\n"
        except Exception:
            result_text += "Log Probability P(Sentence): (error)\n"
        result_text += "\nExpected Counts (marginal γ) for pieces:\n"
        # Aggregate gamma by token type
        gamma_by_piece = {}
        for pos, tokens in enumerate(self.fb_model.lattice.tokens_from_pos):
            for token in tokens:
                gamma = getattr(token, 'gamma', 0.0)
                gamma_by_piece[token.text] = gamma_by_piece.get(token.text, 0.0) + gamma
        sorted_pieces = sorted(gamma_by_piece.items(), key=lambda item: item[1], reverse=True)
        for piece_str, exp_count in sorted_pieces:
            if exp_count > 1e-6:
                result_text += f"  '{piece_str}': {exp_count:.4f}\n"
        result_text += "\nOther pieces (γ ≈ 0):\n"
        for piece_str, exp_count in sorted_pieces:
            if exp_count <= 1e-6:
                result_text += f"  '{piece_str}': {exp_count:.4f}\n"
        # Viterbi path
        try:
            viterbi_path, viterbi_score = self.fb_model.lattice.viterbi()
            result_text += "\nViterbi Path (Best single segmentation):\n"
            if viterbi_path:
                path_pieces = [f"'{token.text}'" for token in viterbi_path]
                result_text += f"  {' + '.join(path_pieces)} (Score: {viterbi_score:.4f})\n"
            else:
                result_text += "  No path found.\n"
        except Exception as e:
            result_text += f"\nViterbi Path: Error computing path - {e}\n"
        self.results_text.setPlainText(result_text)

    def on_node_selected(self, token_info):
        """Update the Node Details panel when a node is selected."""
        if not token_info or not self.fb_model or not self.fb_model.lattice:
            self.detail_text.setPlainText("No node selected or model not ready.")
            return
        token, pos = token_info
        piece_str = token.text
        detail = f"--- Details for Token '{piece_str}' [{pos}-{pos + len(token.text)}) ---\n\n"
        detail += f"Token ID: {token.id}\n"
        detail += f"Log Score (log P('{piece_str}')): {token.log_prob:.4f}\n\n"
        detail += "--- Marginal Probability (Gamma for THIS token occurrence) ---\n"
        gamma_val = getattr(token, 'gamma', 0.0)
        if gamma_val > 0 and gamma_val < 1e-6:
            detail += f"γ(token) = {gamma_val:.4e} (Very small)\n"
        else:
            detail += f"γ(token) = {gamma_val:.4f}\n"
        # Show total expected count for this piece
        gamma_by_piece = 0.0
        for p, tokens in enumerate(self.fb_model.lattice.tokens_from_pos):
            for t in tokens:
                if t.text == piece_str:
                    gamma_by_piece += getattr(t, 'gamma', 0.0)
        detail += f"\n--- Total Expected Count for Piece '{piece_str}' ---\n"
        detail += f"Sum of γ(token) over ALL occurrences for '{piece_str}': {gamma_by_piece:.4f}\n"
        detail += "(This is what contributes to the piece's score update in the M-step.)\n"
        self.detail_text.setPlainText(detail)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Set a default font for better readability
    app.setFont(QtGui.QFont("Arial", 10))
    window = ForwardBackwardVisualizer()
    window.show()
    sys.exit(app.exec())
