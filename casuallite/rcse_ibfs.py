import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, chi2_contingency
from itertools import combinations
from typing import List, Tuple, Dict, Optional

# Suppress all warnings for a cleaner output, as specified in the original code.
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------------------------------------------------------
# 1. Revised Sparse Autoencoder with Deeper Architecture
# -----------------------------------------------------------------------------
class SparseAutoencoder(nn.Module):
    """
    A sparse autoencoder for feature selection with a configurable, deeper architecture.
    """
    def __init__(self, input_dim: int, bottleneck_dim: int, hidden_dims: Optional[List[int]] = None):
        """
        Initializes the autoencoder with a configurable number of hidden layers.
        
        Args:
            input_dim (int): The number of input features.
            bottleneck_dim (int): The dimension of the bottleneck layer.
            hidden_dims (List[int], optional): A list of integers specifying the size
                                               of each hidden layer. Defaults to None (linear).
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = []
            
        # Build the encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, bottleneck_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build the decoder
        decoder_layers = []
        in_dim = bottleneck_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the autoencoder.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_model(self, X: np.ndarray, epochs: int = 200, learning_rate: float = 0.003, l1_lambda: float = 0.005):
        """
        Trains the autoencoder model using MSE loss and L1 regularization.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self(X_tensor)
            loss = criterion(outputs, X_tensor)
            
            # L1 penalty on the encoder's weights
            l1_penalty = sum(torch.norm(p, 1) for p in self.encoder.parameters())
            total_loss = loss + l1_lambda * l1_penalty
            
            total_loss.backward()
            optimizer.step()

    def get_feature_importances(self) -> np.ndarray:
        """
        Calculates feature importances based on the L1 norms of the first encoder layer's weights.
        """
        # The first layer of the encoder has the weights we need
        first_layer_weights = self.encoder[0].weight.data.abs()
        importances = first_layer_weights.sum(dim=0).numpy()
        return importances

# -----------------------------------------------------------------------------
# 2. Revised Information Bottleneck Feature Selection (IBFS)
# -----------------------------------------------------------------------------
def information_bottleneck_feature_selection(
    X: np.ndarray, 
    beta: float, 
    min_features: int = 5, 
    log_names: Optional[List[str]] = None,
    bottleneck_dim: Optional[int] = None,
    hidden_dims: Optional[List[int]] = None
) -> Tuple[List[int], Dict[int, float]]:
    """
    Performs Information Bottleneck-based feature selection.
    
    Args:
        X (np.ndarray): The input data matrix.
        beta (float): A factor to determine the bottleneck dimension.
        min_features (int): Minimum number of features to select.
        log_names (List[str], optional): Feature names for logging.
        bottleneck_dim (int, optional): Explicitly set the bottleneck dimension.
                                        If None, it's calculated automatically.
        hidden_dims (List[int], optional): A list of hidden layer dimensions for the autoencoder.
        
    Returns:
        Tuple[List[int], Dict[int, float]]: A tuple containing the sorted list of selected
                                             feature indices and a dictionary of their importances.
    """
    n_samples, p = X.shape
    
    if bottleneck_dim is None:
        bottleneck_dim = max(min_features, min(p, int(p * beta)))
        
    autoencoder = SparseAutoencoder(p, bottleneck_dim, hidden_dims)
    autoencoder.train_model(X)
    importances = autoencoder.get_feature_importances()
    
    selected_indices = np.argsort(importances)[-bottleneck_dim:][::-1]
    
    print(f"\n✅ Feature selection complete. Selected {len(selected_indices)} features.")
    
    if log_names:
        print("Top selected features:")
        for idx in selected_indices:
            print(f"  ➤ {log_names[idx]} (importance = {importances[idx]:.4f})")
    
    # The dictionary maps original feature indices to their importance scores
    importance_dict = {idx: importances[idx] for idx in selected_indices}
    
    return sorted(selected_indices.tolist()), importance_dict

# -----------------------------------------------------------------------------
# 3. Revised Conditional Independence Test
# -----------------------------------------------------------------------------
def conditional_independence_test(
    X: np.ndarray, 
    i: int, 
    j: int, 
    cond_set: List[int],
    test_type: str = "pearsonr"
) -> float:
    """
    Tests for conditional independence between variables i and j given a set of conditioning variables.
    
    Args:
        X (np.ndarray): The data matrix.
        i (int): Index of the first variable.
        j (int): Index of the second variable.
        cond_set (List[int]): List of indices for the conditioning variables.
        test_type (str): The type of independence test to perform.
                         'pearsonr' for continuous data, 'chi-squared' for discrete.
                         
    Returns:
        float: The p-value of the conditional independence test.
    """
    X_i = X[:, i]
    X_j = X[:, j]

    try:
        if test_type == "pearsonr":
            if not cond_set:
                _, pval = pearsonr(X_i, X_j)
            else:
                model_i = LinearRegression().fit(X[:, cond_set], X_i)
                model_j = LinearRegression().fit(X[:, cond_set], X_j)
                res_i = X_i - model_i.predict(X[:, cond_set])
                res_j = X_j - model_j.predict(X[:, cond_set])
                _, pval = pearsonr(res_i, res_j)
        
        elif test_type == "chi-squared":
            # This requires data to be binned or categorized for chi-squared.
            # For simplicity, we'll assume the input is already categorical.
            # A more robust solution would involve binning continuous data.
            # Here we are just creating a contingency table.
            contingency_table = pd.crosstab(X_i, X_j)
            _, pval, _, _ = chi2_contingency(contingency_table)
            
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
            
    except np.linalg.LinAlgError:
        # Catch specific errors related to singular matrices (multicollinearity)
        warnings.warn(f"Warning: Multicollinearity or numerical issue detected for variables ({i}, {j}) given {cond_set}. Setting p-value to 1.0.", RuntimeWarning)
        pval = 1.0
    except ValueError as e:
        warnings.warn(f"Value error during independence test: {e}. Setting p-value to 1.0.", RuntimeWarning)
        pval = 1.0
    except Exception as e:
        # A more specific catch for other potential issues
        warnings.warn(f"An unexpected error occurred during independence test: {e}. Setting p-value to 1.0.", RuntimeWarning)
        pval = 1.0
        
    return pval

# -----------------------------------------------------------------------------
# 4. Revised Alpha Adjustment
# -----------------------------------------------------------------------------
def adjust_alpha(
    alpha_init: float, 
    p_selected: int, 
    cond_size: int,
    l: int,
    correction_method: str = "custom"
) -> float:
    """
    Adjusts the significance level (alpha) based on the correction method.
    
    Args:
        alpha_init (float): The initial alpha level.
        p_selected (int): Number of features selected.
        cond_size (int): Size of the conditioning set.
        l (int): The current size of the conditioning set being tested.
        correction_method (str): 'custom' or 'bonferroni'.
        
    Returns:
        float: The adjusted alpha level.
    """
    if correction_method == "custom":
        penalty = cond_size / p_selected
        adjusted = min(alpha_init * (1 + penalty), 1.0)
        return adjusted
    
    elif correction_method == "bonferroni":
        # Calculate the number of comparisons being made
        n_pairs = p_selected * (p_selected - 1) // 2
        n_comparisons = len(list(combinations(range(p_selected - 2), l))) * n_pairs
        if n_comparisons == 0:
            n_comparisons = 1 # Avoid division by zero
        adjusted = alpha_init / n_comparisons
        return adjusted
        
    else:
        raise ValueError("Invalid correction_method. Choose 'custom' or 'bonferroni'.")

# -----------------------------------------------------------------------------
# 5. RCSE-IBFS Main Algorithm with Enhanced Features
# -----------------------------------------------------------------------------
def RCSE_IBFS(
    X: np.ndarray, 
    beta: float = 0.9, 
    alpha_init: float = 0.15, 
    max_cond_set_size: int = 2,
    feature_names: Optional[List[str]] = None,
    keep_isolated_nodes: bool = False,
    test_type: str = "pearsonr",
    correction_method: str = "custom",
    bottleneck_dim: Optional[int] = None,
    hidden_dims: Optional[List[int]] = None
) -> Tuple[nx.Graph, Dict[int, float]]:
    """
    Revised Causal Skeleton Estimation with Information Bottleneck Feature Selection.
    
    Args:
        X (np.ndarray): The input data matrix.
        beta (float): Parameter for the information bottleneck.
        alpha_init (float): Initial significance level for conditional independence tests.
        max_cond_set_size (int): Maximum size of conditioning set to test.
        feature_names (List[str], optional): List of feature names.
        keep_isolated_nodes (bool): If True, retains nodes with no connections in the final graph.
        test_type (str): The type of independence test to perform ('pearsonr' or 'chi-squared').
        correction_method (str): The alpha adjustment method ('custom' or 'bonferroni').
        bottleneck_dim (int, optional): Explicitly set the bottleneck dimension for IBFS.
        hidden_dims (List[int], optional): A list of hidden layer dimensions for the autoencoder.
        
    Returns:
        nx.Graph: The recovered causal skeleton graph.
    """
    
    # 7. Add Input Validation
    if np.any(np.isnan(X)):
        raise ValueError("Input data contains missing values (NaN). Please handle them before running the algorithm.")
    n, p = X.shape
    if n < p:
        warnings.warn("Number of samples (n) is less than number of features (p). This may lead to unstable results.", RuntimeWarning)
    if n < 50:
        warnings.warn("Small sample size (n < 50) detected. Results may be unreliable.", RuntimeWarning)
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selected, importances = information_bottleneck_feature_selection(
        X_scaled, beta, log_names=feature_names, bottleneck_dim=bottleneck_dim, hidden_dims=hidden_dims
    )
    
    if not selected:
        print("⚠️ No features selected. Returning empty graph.")
        return nx.Graph(), {}

    # --- NEW: Correlation-based feature pruning ---
    selected, importances = remove_highly_correlated_features(
        X, selected, importances, feature_names=feature_names
    )
    # ----------------------------------------------

    X_sel = X_scaled[:, selected]
    p_sel = len(selected)
    print(f"\nRunning PC on {p_sel} selected features...")

    skeleton = nx.complete_graph(p_sel)
    for l in range(max_cond_set_size + 1):
        edges = list(skeleton.edges())
        for i, j in edges:
            if not skeleton.has_edge(i, j):
                continue
            neighbors = list(set(skeleton.neighbors(i)).union(skeleton.neighbors(j)) - {i, j})
            for cond in combinations(neighbors, l):
                alpha = adjust_alpha(alpha_init, p_sel, len(cond), l, correction_method)
                pval = conditional_independence_test(X_sel, i, j, list(cond), test_type)
                if pval > alpha:
                    skeleton.remove_edge(i, j)
                    break

    # Map back to original feature indices
    G = nx.Graph()
    G.add_nodes_from(range(p))
    for u, v in skeleton.edges():
        G.add_edge(selected[u], selected[v])

    # 6. Add option to retain isolated nodes
    if not keep_isolated_nodes:
        isolated_nodes = [n for n in G.nodes() if G.degree[n] == 0]
        if isolated_nodes:
            print(f"ℹ️ Removing {len(isolated_nodes)} isolated nodes.")
        G.remove_nodes_from(isolated_nodes)
    
    # Return the graph and the importance scores
    return G, importances

# -----------------------------------------------------------------------------
# 5.1 New Function: Correlation-based Pruning
# -----------------------------------------------------------------------------
def remove_highly_correlated_features(
    X: np.ndarray,
    selected_indices: List[int],
    importances: Dict[int, float],
    correlation_threshold: float = 0.9,
    feature_names: Optional[List[str]] = None
) -> Tuple[List[int], Dict[int, float]]:
    """
    Removes one of a pair of highly correlated features, keeping the one with higher importance.
    
    Args:
        X (np.ndarray): The full data matrix (unscaled).
        selected_indices (List[int]): The list of indices of features selected by IBFS.
        importances (Dict[int, float]): A dictionary mapping feature indices to their importance scores.
        correlation_threshold (float): The threshold above which features are considered highly correlated.
        feature_names (List[str], optional): List of feature names for logging.
        
    Returns:
        Tuple[List[int], Dict[int, float]]: Updated list of selected features and importance dictionary.
    """
    # Create a DataFrame for easy correlation calculation
    selected_df = pd.DataFrame(X[:, selected_indices])
    correlation_matrix = selected_df.corr().abs()
    
    to_remove_indices = set()
    
    # Use the original feature indices for mapping
    mapping = {idx: original_idx for idx, original_idx in enumerate(selected_indices)}
    
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            if correlation_matrix.iloc[i, j] > correlation_threshold:
                # Get the original feature indices and importances
                idx_i = mapping[i]
                idx_j = mapping[j]
                
                imp_i = importances.get(idx_i, 0)
                imp_j = importances.get(idx_j, 0)
                
                # Keep the one with the higher importance
                if imp_i > imp_j:
                    to_remove_indices.add(idx_j)
                    if feature_names:
                        print(f"✂️ Removing redundant feature '{feature_names[idx_j]}' (correlated with '{feature_names[idx_i]}')")
                else:
                    to_remove_indices.add(idx_i)
                    if feature_names:
                        print(f"✂️ Removing redundant feature '{feature_names[idx_i]}' (correlated with '{feature_names[idx_j]}')")

    # Filter the selected indices and importances
    new_selected_indices = [idx for idx in selected_indices if idx not in to_remove_indices]
    new_importances = {idx: imp for idx, imp in importances.items() if idx in new_selected_indices}
    
    print(f"\n✨ After correlation filtering, {len(new_selected_indices)} features remain.")
    
    return sorted(new_selected_indices), new_importances

# -----------------------------------------------------------------------------
# 8. Revised Graph Visualization
# -----------------------------------------------------------------------------
def plot_graph(
    G: nx.Graph, 
    labels: List[str], 
    title: str = "Recovered Causal Skeleton", 
    node_importance: Optional[Dict[int, float]] = None,
    node_size_scale: float = 1000,
    edge_width: float = 2
):
    """
    Plots the graph with customizable visualization options.
    
    Args:
        G (nx.Graph): The graph to plot.
        labels (List[str]): List of labels for all features.
        title (str): The title of the plot.
        node_importance (Dict[int, float], optional): A dictionary mapping node indices
                                                       to importance scores. Used for sizing nodes.
        node_size_scale (float): A scaling factor for node sizes.
        edge_width (float): The width of the edges.
    """
    plt.figure(figsize=(12, 8))
    
    active_labels = {i: labels[i] for i in G.nodes()}
    pos = nx.spring_layout(G, seed=42)
    
    node_sizes = [node_size_scale] * len(G.nodes())
    if node_importance:
        # Filter importance to only include nodes present in the graph G
        filtered_importance = {node: importance for node, importance in node_importance.items() if node in G.nodes()}
        
        max_importance = max(filtered_importance.values()) if filtered_importance else 0
        if max_importance > 0:
            node_sizes = [filtered_importance.get(node, 0) / max_importance * node_size_scale + 200 for node in G.nodes()]

    node_colors = sns.color_palette("viridis", len(G.nodes()))
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.6)
    nx.draw_networkx_labels(G, pos, labels=active_labels, font_size=10, font_weight='bold')
    
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()