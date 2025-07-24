# --- model.py_integration_gui.py ---
import sys
import math
from collections import defaultdict

# --- Import the Lattice and TrainerModel classes ---
# Ensure model.py is in the same directory or on the Python path
from py_unigram.qwen.model import Lattice, TrainerModel, Node

# --- PyQt6 Imports ---
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QPen, QBrush, QFont, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QMessageBox,
    QGroupBox, QSplitter, QFrame, QGraphicsScene, QGraphicsView,
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsLineItem
)

# --- GUI Logic ---

class LatticeNodeWidget(QGraphicsRectItem):
    """Represents a single node in the lattice visualization."""
    def __init__(self, node: Node, trainer_model: TrainerModel):
        super().__init__()
        self.node = node
        self.trainer_model = trainer_model
        self.setRect(0, 0, 120, 70) # Slightly larger for better text fit
        self.initial_brush = QBrush(QColor("lightgray"))
        self.selected_brush = QBrush(QColor("lightblue"))
        self.setBrush(self.initial_brush)
        self.setPen(QPen(Qt.GlobalColor.black, 1))
        
        # Create text item as a child for easier positioning
        self.text_item = QGraphicsTextItem(parent=self)
        self.text_item.setDefaultTextColor(Qt.GlobalColor.black)
        self.text_item.setPos(5, 5)
        self.update_display()

    def update_display(self):
        """Updates the text based on node data."""
        try:
            piece_str = self.trainer_model[self.node.piece_id][0]
        except (IndexError, KeyError):
            piece_str = f"<unk:{self.node.piece_id}>"
            
        gamma_text = f"γ:{self.node.gamma:.3f}" if hasattr(self.node, 'gamma') and self.node.gamma > 1e-6 else "γ:0.000"
        alpha_text = f"α:{self.node.backtrace_score:.2f}" # Use backtrace_score as alpha proxy
        beta_text = f"β:{self.node.beta_score:.2f}"

        self.text_item.setPlainText(
            f"'{piece_str}'\n"
            f"[{self.node.pos}-{self.node.pos + self.node.length})\n"
            f"{gamma_text}\n"
            # f"{alpha_text} {beta_text}" # Optional: show alpha/beta on node if space allows
        )
        # Adjust rect size based on text if needed (basic attempt)
        # This can be refined further
        text_bounding_rect = self.text_item.boundingRect()
        if text_bounding_rect.width() > self.rect().width() - 10:
             self.setRect(0, 0, text_bounding_rect.width() + 10, self.rect().height())
        if text_bounding_rect.height() > self.rect().height() - 10:
             self.setRect(0, 0, self.rect().width(), text_bounding_rect.height() + 10)

    def set_selected(self, is_selected):
        """Visually indicate selection."""
        if is_selected:
            self.setBrush(self.selected_brush)
        else:
            self.setBrush(self.initial_brush)
        # Force an update
        self.update()


class LatticeView(QGraphicsView):
    """Visualizes the lattice with nodes and transitions."""
    node_selected = QtCore.pyqtSignal(object) # Emits Lattice.Node or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lattice = None
        self.trainer_model = None
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.nodes_widgets = {}  # Key: Lattice.Node, Value: LatticeNodeWidget
        self.selected_node_widget = None
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

    def update_view(self, lattice: Lattice, trainer_model: TrainerModel):
        """Rebuilds the lattice visualization."""
        self.lattice = lattice
        self.trainer_model = trainer_model
        self.scene.clear()
        self.nodes_widgets = {}
        self.selected_node_widget = None

        if not self.lattice or not self.trainer_model:
            return

        N = len(self.lattice.chars)
        if N == 0:
            return

        # --- Create nodes ---
        max_y = 0
        for pos_nodes in self.lattice.nodes:
            for node in pos_nodes:
                node_widget = LatticeNodeWidget(node, self.trainer_model)
                self.nodes_widgets[node] = node_widget
                self.scene.addItem(node_widget)
                # Position nodes: x by start pos, y by end pos for a triangular look
                x = node.pos * 130
                y = (node.pos + node.length) * 80
                if y > max_y: max_y = y
                node_widget.setPos(x, y)

        # --- Draw arrows for transitions ---
        pen = QPen(QColor("blue"), 1.5, Qt.PenStyle.SolidLine)
        for pos_nodes in self.lattice.nodes:
            for node in pos_nodes:
                start_widget = self.nodes_widgets.get(node)
                if not start_widget: continue
                end_pos = node.pos + node.length
                for next_node in self.lattice.nodes[end_pos]:
                    end_widget = self.nodes_widgets.get(next_node)
                    if not end_widget: continue
                    
                    # Arrow from bottom center of start to top center of end
                    start_point_scene = start_widget.mapToScene(start_widget.boundingRect().center().x(), start_widget.boundingRect().bottom())
                    end_point_scene = end_widget.mapToScene(end_widget.boundingRect().center().x(), end_widget.boundingRect().top())
                    
                    arrow = QGraphicsLineItem(
                        start_point_scene.x(), start_point_scene.y(),
                        end_point_scene.x(), end_point_scene.y()
                    )
                    arrow.setPen(pen)
                    self.scene.addItem(arrow)

        # --- Set scene size ---
        scene_width = max(800, (N + 2) * 130)
        scene_height = max(600, max_y + 200) # Add padding
        self.scene.setSceneRect(0, 0, scene_width, scene_height)
        # print(f"Scene size set to: {scene_width} x {scene_height}") # Debug

    def mousePressEvent(self, event):
        """Handle clicks to select nodes."""
        # Deselect previous node
        if self.selected_node_widget:
            self.selected_node_widget.set_selected(False)
            self.selected_node_widget = None

        # Check if an item was clicked
        clicked_item = self.itemAt(event.pos())
        node_to_emit = None

        if clicked_item:
            # Find the top-level LatticeNodeWidget
            top_item = clicked_item
            while top_item and not isinstance(top_item, LatticeNodeWidget):
                top_item = top_item.parentItem() if hasattr(top_item, 'parentItem') else None

            if isinstance(top_item, LatticeNodeWidget):
                top_item.set_selected(True)
                self.selected_node_widget = top_item
                node_to_emit = top_item.node

        # Emit signal even if deselected (None)
        self.node_selected.emit(node_to_emit)
        # Call parent *after* our logic to ensure normal behavior
        super().mousePressEvent(event)


class SimpleUnigramFBModel:
    """Coordinator for the Forward-Backward process using model.py."""
    def __init__(self, vocab_scores: dict): # vocab_scores: {piece_str: log_score}
        # Convert log scores to frequencies for TrainerModel
        self.pretokens = {piece: math.exp(score) for piece, score in vocab_scores.items()}
        self.sentence = ""
        self.trainer_model: TrainerModel = None
        self.lattice: Lattice = None
        self.log_Z = float('-inf')
        self.expected_counts = [] # From populate_marginal

    def run_full_algorithm(self, sentence: str):
        """Runs the entire process."""
        self.sentence = sentence
        if not self.sentence:
            return

        # 1. Create TrainerModel
        self.trainer_model = TrainerModel(self.pretokens)

        # 2. Create and populate Lattice
        self.lattice = Lattice(self.sentence)
        self.trainer_model.populate(self.lattice)

        # 3. Run Forward-Backward (E-step calculation for this sentence)
        # Use freq=1.0 for calculating marginals. The returned value is logZ.
        self.expected_counts = [0.0] * len(self.trainer_model)
        self.log_Z = self.lattice.populate_marginal(1.0, self.expected_counts)

        # 4. Calculate gamma for each node
        # gamma(node) = P(node used in sentence) = exp(alpha[pos] + log_prob + beta[end] - Z)
        # Note: populate_marginal stores alpha in self.lattice.alpha and beta in self.lattice.beta
        sentence_len = len(self.lattice.chars)
        if self.log_Z != float('-inf') and sentence_len > 0:
            for pos in range(sentence_len):
                for node in self.lattice.nodes[pos]:
                    end_pos = pos + node.length
                    # Ensure indices are valid
                    if pos < len(self.lattice.alpha) and end_pos < len(self.lattice.beta):
                        log_alpha = self.lattice.alpha[pos]
                        log_prob = node.log_prob
                        log_beta = self.lattice.beta[end_pos]
                        if log_alpha != float('-inf') and log_beta != float('-inf'):
                            log_gamma = log_alpha + log_prob + log_beta - self.log_Z
                            # Clamp to prevent underflow
                            node.gamma = math.exp(log_gamma) if log_gamma > -100 else 0.0
                        else:
                            node.gamma = 0.0
                    else:
                        node.gamma = 0.0
        else:
            # If Z is -inf, no valid path, so gamma is 0 for all nodes
            for pos_nodes in self.lattice.nodes:
                for node in pos_nodes:
                    node.gamma = 0.0

        # Trigger Viterbi computation for alpha/beta display if needed elsewhere
        # (populate_marginal might have already done parts of it)
        # self.lattice.viterbi() # Not strictly necessary for gamma, but good for consistency


class ForwardBackwardVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Educational Forward-Backward for Unigram LM (model.py)")
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
        default_vocab = "h -23.5\ne -0.4\nl -0.6\no -0.5\nhe -1.2\nel -1.3\nll -1.4\nlo -1.1\nhel -2.0\nell -2.1\nllo -2.2\nhello -3.0"
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
            self.fb_model = SimpleUnigramFBModel(vocab_dict)
            self.fb_model.run_full_algorithm(sentence)
            
            self.lattice_view.update_view(self.fb_model.lattice, self.fb_model.trainer_model)
            self.update_overall_results()
            self.detail_text.setPlainText("Click on a node in the lattice to see its details.")

        except Exception as e:
            QMessageBox.critical(self, "Algorithm Error", f"An error occurred: {e}")
            import traceback
            traceback.print_exc() # For debugging

    def update_overall_results(self):
        if not self.fb_model or not self.fb_model.lattice:
            self.results_text.setPlainText("")
            return

        result_text = f"Sentence: '{self.fb_model.sentence}'\n"
        if self.fb_model.log_Z == float('-inf'):
            result_text += "Log Probability P(Sentence): -inf\n"
            result_text += "Probability P(Sentence): 0.0\n"
        else:
            result_text += f"Log Probability P(Sentence): {self.fb_model.log_Z:.4f}\n"
            result_text += f"Probability P(Sentence): {math.exp(self.fb_model.log_Z):.4e}\n"

        result_text += "\nExpected Counts (freq=1.0 * γ) for pieces:\n"
        # Sort by expected count descending
        if self.fb_model.trainer_model and self.fb_model.expected_counts:
            piece_scores = list(self.fb_model.trainer_model.pieces)
            piece_data = list(zip(piece_scores, self.fb_model.expected_counts))
            sorted_pieces = sorted(piece_data, key=lambda item: item[1], reverse=True)
            for (piece_str, _), exp_count in sorted_pieces:
                if exp_count > 1e-6: # Only show significant ones
                    result_text += f"  '{piece_str}': {exp_count:.4f}\n"
            # Show zero or near-zero ones too for completeness
            result_text += "\nOther pieces (γ ≈ 0):\n"
            for (piece_str, _), exp_count in sorted_pieces:
                if exp_count <= 1e-6:
                    result_text += f"  '{piece_str}': {exp_count:.4f}\n"

        # Also show Viterbi path
        try:
            viterbi_path, viterbi_score = self.fb_model.lattice.viterbi()
            result_text += f"\nViterbi Path (Best single segmentation):\n"
            if viterbi_path:
                path_pieces = []
                for node in viterbi_path:
                    try:
                        piece_str = self.fb_model.trainer_model[node.piece_id][0]
                    except (IndexError, KeyError):
                        piece_str = f"<unk:{node.piece_id}>"
                    path_pieces.append(f"'{piece_str}'")
                result_text += f"  {' + '.join(path_pieces)} (Score: {viterbi_score:.4f})\n"
            else:
                result_text += "  No path found.\n"
        except Exception as e:
            result_text += f"\nViterbi Path: Error computing path - {e}\n"

        self.results_text.setPlainText(result_text)

    def on_node_selected(self, node: Node):
        """Update the Node Details panel when a node is selected."""
        if not node or not self.fb_model or not self.fb_model.trainer_model:
            self.detail_text.setPlainText("No node selected or model not ready.")
            return

        try:
            piece_str = self.fb_model.trainer_model[node.piece_id][0]
        except (IndexError, KeyError):
            piece_str = f"<unk:{node.piece_id}>"

        detail = f"--- Details for Node '{piece_str}' [{node.pos}-{node.pos + node.length}) ---\n\n"
        detail += f"Piece ID: {node.piece_id}\n"
        detail += f"Log Score (log P('{piece_str}')): {node.log_prob:.4f}\n\n"

        detail += "--- Viterbi Alpha (Best path TO this node's start) ---\n"
        # Use backtrace_score as it represents the best score *to* this node
        if node.backtrace_score == float('-inf'):
            detail += "Alpha (backtrace_score): -inf\n"
            detail += "Explanation: This node is not reachable via the best Viterbi path.\n"
        else:
            detail += f"Alpha (backtrace_score): {node.backtrace_score:.4f}\n"
            # Optionally, try to trace back the Viterbi path segment leading to this node
            # This is complex, so we'll just note it's the score.

        detail += "\n--- N-best Beta (Best path FROM this node's end) ---\n"
        if node.beta_score == float('-inf'):
            detail += "Beta (beta_score): -inf\n"
            detail += "Explanation: This node does not lead to a valid completion via the best path.\n"
        else:
            detail += f"Beta (beta_score): {node.beta_score:.4f}\n"

        detail += f"\n--- Marginal Probability (Gamma for THIS node) ---\n"
        detail += f"Gamma γ(node) = P(using '{piece_str}' at [{node.pos}-{node.pos + node.length}) | sentence)\n"
        if hasattr(node, 'gamma'):
            gamma_val = node.gamma
            if gamma_val > 0 and gamma_val < 1e-6:
                detail += f"γ(node) = {gamma_val:.4e} (Very small)\n"
            else:
                detail += f"γ(node) = {gamma_val:.4f}\n"
        else:
            detail += f"γ(node) = 0.0 (Not calculated)\n"

        if self.fb_model.expected_counts and node.piece_id < len(self.fb_model.expected_counts):
            total_gamma_for_piece = self.fb_model.expected_counts[node.piece_id]
            detail += f"\n--- Total Expected Count for Piece '{piece_str}' ---\n"
            detail += f"Sum of γ(node) over ALL nodes for '{piece_str}': {total_gamma_for_piece:.4f}\n"
            detail += f"(This is what contributes to the piece's score update in the M-step.)\n"

        self.detail_text.setPlainText(detail)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Set a default font for better readability
    app.setFont(QtGui.QFont("Arial", 10)) 
    window = ForwardBackwardVisualizer()
    window.show()
    sys.exit(app.exec())
