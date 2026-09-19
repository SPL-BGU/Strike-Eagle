import warnings

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import ConvergenceWarning
from lineartree import LinearTreeRegressor

# Noisy on small leaves / tiny dual gaps; max_iter=10000 is already set on Lasso/ElasticNet.
warnings.filterwarnings('ignore', category=ConvergenceWarning)


# ==============================================================================
# VELOCITY RATIO FEATURE FOR ABLATION STUDY
# ==============================================================================

def compute_velocity_ratio(v_x, v_y, epsilon=1e-6):
    """
    Compute normalized velocity ratio (impact angle proxy).
    
    Returns |v_y| / (|v_x| + |v_y| + epsilon) to avoid division by zero.
    Range: 0 (horizontal impact) to 1 (vertical impact)
    
    This feature captures the impact angle which may affect bounce behavior:
    - Shallow impacts (ratio near 0): More sliding, less bounce
    - Steep impacts (ratio near 1): More bouncing, less sliding
    
    Parameters:
        v_x: Horizontal velocity component
        v_y: Vertical velocity component  
        epsilon: Small value to prevent division by zero
        
    Returns:
        float: Normalized velocity ratio in range [0, 1]
    """
    total = abs(v_x) + abs(v_y) + epsilon
    return abs(v_y) / total


def compute_angle_trig_features(v_x, v_y, epsilon=1e-6):
    """
    Compute trigonometric features based on velocity angle (atan2(v_y, v_x)).
    
    These features capture the impact angle in different representations:
    - cos(angle): Horizontal component normalized by speed
    - sin(angle): Vertical component normalized by speed  
    - tan(angle): Ratio v_y/v_x (clamped to avoid infinity)
    
    Parameters:
        v_x: Horizontal velocity component
        v_y: Vertical velocity component
        epsilon: Small value to prevent division by zero
        
    Returns:
        tuple: (cos_angle, sin_angle, tan_angle)
    """
    # Compute speed (magnitude of velocity)
    speed = np.sqrt(v_x**2 + v_y**2)
    
    if speed < epsilon:
        # Near-zero velocity: return neutral values
        return (1.0, 0.0, 0.0)
    
    # cos and sin are normalized velocity components
    cos_angle = v_x / speed
    sin_angle = v_y / speed
    
    # tan with clamping to avoid extreme values
    if abs(v_x) < epsilon:
        # Near-vertical: clamp tan to a large but finite value
        tan_angle = np.sign(v_y) * 100.0 if abs(v_y) > epsilon else 0.0
    else:
        tan_angle = v_y / v_x
        # Clamp to reasonable range
        tan_angle = np.clip(tan_angle, -100.0, 100.0)
    
    return (cos_angle, sin_angle, tan_angle)


def compute_kinetic_features(v_x, v_y):
    """
    Compute kinetic energy-related features from velocity components.
    
    These features capture energy and momentum characteristics:
    - speed (v): sqrt(v_x² + v_y²) - total velocity magnitude
    - v_x²: squared horizontal velocity (proportional to horizontal kinetic energy)
    - v_y²: squared vertical velocity (proportional to vertical kinetic energy)
    
    Physics motivation:
    - Kinetic energy = 0.5 * m * v² = 0.5 * m * (v_x² + v_y²)
    - Collision outcomes often depend on energy, not just velocity
    - Squared terms capture non-linear energy relationships
    
    Parameters:
        v_x: Horizontal velocity component
        v_y: Vertical velocity component
        
    Returns:
        tuple: (speed, v_x_squared, v_y_squared)
    """
    speed = np.sqrt(v_x**2 + v_y**2)
    v_x_squared = v_x**2
    v_y_squared = v_y**2
    
    return (speed, v_x_squared, v_y_squared)


# ==============================================================================
# REGULARIZATION OPTIONS FOR EVENT LEARNING
# ==============================================================================
#
# Loss Functions:
#
# 1. No Regularization (OLS):
#    L = Σ(y - ŷ)²
#
# 2. L1 Regularization (Lasso):
#    L = Σ(y - ŷ)² + α·Σ|βⱼ|
#    → Drives irrelevant coefficients to EXACTLY ZERO (feature selection)
#
# 3. L2 Regularization (Ridge):
#    L = Σ(y - ŷ)² + α·Σβⱼ²
#    → Shrinks ALL coefficients toward zero (prevents large values)
#
# 4. L1 + L2 Regularization (ElasticNet):
#    L = Σ(y - ŷ)² + α·[ρ·Σ|βⱼ| + (1-ρ)/2·Σβⱼ²]
#    → Combines sparsity (L1) with stability (L2)
#
# ==============================================================================


def polyfit_var(X, y):
    """
    Fits a polynomial linear regression model of degree 1 on the input features X and targets y.

    Parameters:
        X (np.ndarray): Input features of shape (n_samples, n_features).
        y (np.ndarray): Target values of shape (n_samples,).

    Returns:
        model (LinearRegression): Trained linear regression model.
    """
    feature_names = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y']

    # Polynomial feature transformation and scaling
    poly = PolynomialFeatures(degree=1,include_bias=False)
    scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    X_poly = poly.fit_transform(X)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    return model


class CARTEventModel:
    """
    CART (Classification and Regression Trees) model for event learning.
    
    Uses DecisionTreeRegressor to learn conditional effects based on pre-event state.
    Automatically selects optimal tree depth via cross-validation.
    
    Key features:
    - CV-based depth selection (tests depths 1-3, capped to prevent overfitting)
    - min_samples_leaf=3 prevents overfitting
    - Generates PDDL conditional effects from tree structure
    - Compatible interface with General models (coef_, intercept_)
    """
    
    # Maximum depth cap to prevent overfitting (reduced from 5 to 3)
    MAX_DEPTH_CAP = 3
    
    def __init__(self, var_name):
        self.var_name = var_name
        self.tree = None
        self.best_depth = None
        self.cv_scores = {}
        
        # For compatibility with existing code (inject_domain_file)
        self.coef_ = np.array([0.0, 0.0, 0.0, 0.0])
        self.intercept_ = 0.0
        self.model_type = 'cart'
    
    def fit(self, X, y):
        """
        Fit the CART model with cross-validation to select optimal depth.
        
        Parameters:
            X: Feature matrix of shape (n_samples, 4) with [x, y, v_x, v_y]
            y: Target values (post-event values)
        
        Returns:
            self
        """
        n_samples = len(y)
        
        # Need minimum samples for meaningful tree
        if n_samples < 3:
            # Fall back to mean prediction (single leaf)
            self.tree = DecisionTreeRegressor(max_depth=1, min_samples_leaf=1)
            self.tree.fit(X, y)
            self.best_depth = 1
            self._set_compatibility_attributes(X, y)
            return self
        
        # Cross-validation to find optimal depth (capped at MAX_DEPTH_CAP to prevent overfitting)
        max_depth_limit = min(self.MAX_DEPTH_CAP + 1, n_samples // 2 + 1)
        max_depths_to_test = range(1, max_depth_limit)
        best_cv_score = float('inf')
        best_depth = 1
        
        for depth in max_depths_to_test:
            # LOO-CV for this depth
            cv_errors = []
            min_leaf = max(1, n_samples // (2 ** depth + 1))  # Adaptive min_samples_leaf
            min_leaf = min(min_leaf, 3)  # Cap at 3
            
            for i in range(n_samples):
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i)
                X_test = X[i:i+1]
                y_test = y[i]
                
                try:
                    tree = DecisionTreeRegressor(
                        max_depth=depth,
                        min_samples_leaf=min_leaf,
                        random_state=42
                    )
                    tree.fit(X_train, y_train)
                    pred = tree.predict(X_test)[0]
                    cv_errors.append((pred - y_test) ** 2)
                except Exception:
                    continue
            
            if len(cv_errors) > 0:
                cv_rmse = np.sqrt(np.mean(cv_errors))
                self.cv_scores[depth] = cv_rmse
                
                if cv_rmse < best_cv_score:
                    best_cv_score = cv_rmse
                    best_depth = depth
        
        # Train final model with best depth
        self.best_depth = best_depth
        min_leaf = max(1, min(3, n_samples // (2 ** best_depth + 1)))
        
        self.tree = DecisionTreeRegressor(
            max_depth=best_depth,
            min_samples_leaf=min_leaf,
            random_state=42
        )
        self.tree.fit(X, y)
        
        self._set_compatibility_attributes(X, y)
        return self
    
    def print_tree_rules(self):
        """
        Print the decision rules learned by the CART tree in a human-readable format.
        """
        if self.tree is None:
            print(f"  [{self.var_name}] No tree fitted yet")
            return
        
        feature_names = ['x', 'y', 'v_x', 'v_y']
        tree_ = self.tree.tree_
        
        print(f"\n  {'='*60}")
        print(f"  CART RULES for {self.var_name}_after (depth={self.best_depth}, leaves={tree_.n_leaves})")
        print(f"  {'='*60}")
        
        rules = self.get_tree_rules()
        
        for i, (conditions, leaf_value) in enumerate(rules):
            if not conditions:
                print(f"  Leaf {i+1}: {self.var_name}_after = {leaf_value:.4f} (default)")
            else:
                cond_strs = []
                for feat_idx, threshold, direction in conditions:
                    feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"f{feat_idx}"
                    cond_strs.append(f"{feat_name} {direction} {threshold:.2f}")
                
                print(f"  Leaf {i+1}: IF {' AND '.join(cond_strs)}")
                print(f"           THEN {self.var_name}_after = {leaf_value:.4f}")
        
        # Also print impact factor interpretation if v_x or v_y splits on velocity
        print(f"\n  Interpretation:")
        for conditions, leaf_value in rules:
            if conditions:
                # Check if split is on v_y (index 3) - related to impact angle
                vy_splits = [c for c in conditions if c[0] == 3]
                vx_splits = [c for c in conditions if c[0] == 2]
                
                if vy_splits or vx_splits:
                    if self.var_name == 'v_x':
                        print(f"    - {leaf_value:.4f}: ", end="")
                        if vy_splits:
                            for _, thresh, direction in vy_splits:
                                if direction == '<':
                                    print(f"shallow impact (v_y < {thresh:.0f})", end=" ")
                                else:
                                    print(f"steep impact (v_y >= {thresh:.0f})", end=" ")
                        print()
        
        print(f"  {'='*60}")
    
    def _set_compatibility_attributes(self, X, y):
        """Set coef_ and intercept_ for compatibility with inject_domain_file."""
        # For single-leaf trees, extract the mean as intercept
        if self.tree.tree_.node_count == 1:
            self.intercept_ = self.tree.tree_.value[0, 0, 0]
            self.coef_ = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            # For multi-leaf trees, set intercept to 0 and coef to indicate CART model
            self.intercept_ = 0.0
            self.coef_ = np.array([0.0, 0.0, 0.0, 0.0])
    
    def predict(self, X):
        """Predict using the trained tree."""
        if self.tree is None:
            return np.zeros(len(X))
        return self.tree.predict(X)
    
    def get_tree_rules(self):
        """
        Extract decision rules from the tree for PDDL generation.
        
        Returns:
            List of tuples: [(conditions, leaf_value), ...]
            where conditions is a list of (feature_idx, threshold, direction)
            and direction is '<' or '>='
        """
        if self.tree is None:
            return []
        
        tree = self.tree.tree_
        rules = []
        
        def recurse(node_id, path):
            # Check if leaf
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Leaf node
                leaf_value = tree.value[node_id, 0, 0]
                rules.append((list(path), leaf_value))
                return
            
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            # Left child: feature < threshold
            left_child = tree.children_left[node_id]
            recurse(left_child, path + [(feature, threshold, '<')])
            
            # Right child: feature >= threshold
            right_child = tree.children_right[node_id]
            recurse(right_child, path + [(feature, threshold, '>=')])
        
        recurse(0, [])
        return rules
    
    def get_n_leaves(self):
        """Return number of leaves in the tree."""
        if self.tree is None:
            return 0
        return self.tree.tree_.n_leaves
    
    def is_single_leaf(self):
        """Check if tree is just a single leaf (no splits)."""
        return self.get_n_leaves() == 1
    
    def get_stats(self):
        """Return statistics about the CART model."""
        return {
            'best_depth': self.best_depth,
            'n_leaves': self.get_n_leaves(),
            'cv_scores': self.cv_scores,
            'is_single_leaf': self.is_single_leaf()
        }


# M5 compares these leaf estimators (alpha / l1_ratio from REGULARIZATION_CONFIG).
# OLS ('none') and Ridge leaf ('l2') training skipped — only L1 + ElasticNet leaves compete for injection.
# M5_LEAF_REG_ORDER = ('none', 'l1', 'l2', 'elasticnet')
M5_LEAF_REG_ORDER = ('l1', 'elasticnet')
# Retrain collision M5 every N completed train levels (samples still accumulated each shot).
COLLISION_RETRAIN_EVERY_N_LEVELS = 10
M5_LEAF_REG_LABELS = {
    'none': 'M5 (no reg / OLS leaves)',
    'l1': 'M5 (L1 / Lasso leaves)',
    'l2': 'M5 (L2 / Ridge leaves)',
    'elasticnet': 'M5 (L1+L2 / ElasticNet leaves)',
}


def _m5_leaf_estimator(leaf_reg='l1', alpha=None, l1_ratio=None):
    """
    Linear model at M5 leaves / depth-0 baseline.
    leaf_reg: 'none' | 'l1' | 'l2' | 'elasticnet'
    """
    cfg = REGULARIZATION_CONFIG
    if alpha is None:
        alpha = float(cfg['alpha'])
    if l1_ratio is None:
        l1_ratio = float(cfg['l1_ratio'])
    if leaf_reg == 'none':
        return LinearRegression()
    if leaf_reg == 'l1':
        return Lasso(alpha=alpha, max_iter=10000)
    if leaf_reg == 'l2':
        return Ridge(alpha=alpha, max_iter=10000)
    if leaf_reg == 'elasticnet':
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    return Lasso(alpha=alpha, max_iter=10000)


class M5ModelTree:
    """
    M5 Model Tree for event learning.
    
    Unlike standard CART which outputs constants at leaves, M5 Model Trees
    output LINEAR MODELS at each leaf, allowing it to learn formulas like:
        v_x_after = 0.65 * v_x_before
    
    Uses linear-tree library's LinearTreeRegressor.
    
    Key features:
    - Leaf regularization: none (OLS), L1, L2, or ElasticNet (see m5_leaf_reg)
    - CV-based depth selection (tests depths 1-2 only; max depth capped at 2)
    - Each leaf contains coefficients for: intercept + x + y + v_x + v_y
    """
    
    MAX_DEPTH_CAP = 1  # Reduced from 2 to prevent overfitting
    
    def __init__(self, var_name, m5_leaf_reg='l1'):
        self.var_name = var_name
        self.m5_leaf_reg = m5_leaf_reg  # 'none', 'l1', 'l2', 'elasticnet'
        self.tree = None
        self.best_depth = None
        self.cv_scores = {}
        self.leaf_models = []  # Store (conditions, linear_model) for each leaf
        
        # For compatibility with existing code
        self.coef_ = np.array([0.0, 0.0, 0.0, 0.0])
        self.intercept_ = 0.0
        self.model_type = 'm5'
    
    def fit(self, X, y):
        """
        Fit the M5 Model Tree with cross-validation to select optimal depth.
        
        Parameters:
            X: Feature matrix of shape (n_samples, 4) with [x, y, v_x, v_y]
            y: Target values (post-event values)
        
        Returns:
            self
        """
        n_samples = len(y)
        
        reg_label = M5_LEAF_REG_LABELS.get(self.m5_leaf_reg, self.m5_leaf_reg)
        print(f"[M5 DEBUG] Fitting {reg_label} with {n_samples} samples")
        
        # Need minimum samples for meaningful tree with linear models
        if n_samples < 4:
            print(f"[M5 DEBUG] Too few samples, using leaf baseline ({self.m5_leaf_reg})")
            self.tree = _m5_leaf_estimator(self.m5_leaf_reg)
            self.tree.fit(X, y)
            self.best_depth = 0
            self.coef_ = self.tree.coef_
            self.intercept_ = self.tree.intercept_
            return self
        
        # Try depths 1 to MAX_DEPTH_CAP (max 2)
        max_depths_to_test = range(1, self.MAX_DEPTH_CAP + 1)
        best_cv_score = float('inf')
        best_depth = 0  # Default to Lasso baseline (no splits)
        
        # Depth 0: global Lasso (L1) baseline for CV comparison
        lr_cv_errors = []
        for i in range(n_samples):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            X_test = X[i:i+1]
            y_test = y[i]
            lr = _m5_leaf_estimator(self.m5_leaf_reg)
            lr.fit(X_train, y_train)
            pred = lr.predict(X_test)[0]
            lr_cv_errors.append((pred - y_test) ** 2)
        
        lr_cv_rmse = np.sqrt(np.mean(lr_cv_errors))
        self.cv_scores[0] = lr_cv_rmse
        best_cv_score = lr_cv_rmse
        print(f"[M5 DEBUG] Depth 0 ({self.m5_leaf_reg}) CV-RMSE: {lr_cv_rmse:.4f}")
        
        for depth in max_depths_to_test:
            cv_errors = []
            # Use small min_samples_leaf to allow splits
            min_leaf = 2
            
            for i in range(n_samples):
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i)
                X_test = X[i:i+1]
                y_test = y[i]
                
                try:
                    tree = LinearTreeRegressor(
                        base_estimator=_m5_leaf_estimator(self.m5_leaf_reg),
                        max_depth=depth,
                        min_samples_leaf=max(3, min_leaf)  # LinearTreeRegressor requires > 2
                    )
                    tree.fit(X_train, y_train)
                    pred = tree.predict(X_test)[0]
                    cv_errors.append((pred - y_test) ** 2)
                except Exception as e:
                    print(f"[M5 DEBUG] Depth {depth}, fold {i} failed: {e}")
                    continue
            
            if len(cv_errors) > 0:
                cv_rmse = np.sqrt(np.mean(cv_errors))
                self.cv_scores[depth] = cv_rmse
                print(f"[M5 DEBUG] Depth {depth} CV-RMSE: {cv_rmse:.4f} ({len(cv_errors)}/{n_samples} folds)")
                
                if cv_rmse < best_cv_score:
                    best_cv_score = cv_rmse
                    best_depth = depth
            else:
                print(f"[M5 DEBUG] Depth {depth} - all folds failed!")
        
        # Train final model with best depth
        self.best_depth = best_depth
        print(f"[M5 DEBUG] Selected depth: {best_depth}")
        
        if best_depth == 0:
            self.tree = _m5_leaf_estimator(self.m5_leaf_reg)
            self.tree.fit(X, y)
            self.coef_ = self.tree.coef_
            self.intercept_ = self.tree.intercept_
        else:
            try:
                self.tree = LinearTreeRegressor(
                    base_estimator=_m5_leaf_estimator(self.m5_leaf_reg),
                    max_depth=best_depth,
                    min_samples_leaf=3  # LinearTreeRegressor requires > 2
                )
                self.tree.fit(X, y)
            except Exception as e:
                print(f"[M5 DEBUG] Final fit failed: {e}, falling back to leaf baseline")
                self.tree = _m5_leaf_estimator(self.m5_leaf_reg)
                self.tree.fit(X, y)
                self.best_depth = 0
                self.coef_ = self.tree.coef_
                self.intercept_ = self.tree.intercept_
        
        # Extract leaf models for printing
        self._extract_leaf_models(X, y)
        
        return self
    
    def _extract_leaf_models(self, X, y):
        """Extract the linear models from each leaf."""
        self.leaf_models = []
        
        if not hasattr(self.tree, 'summary'):
            return
        
        try:
            summary = self.tree.summary()
            # summary contains info about splits and leaves
            self.tree_summary = summary
        except Exception:
            pass
    
    def predict(self, X):
        """Predict using the trained tree."""
        if self.tree is None:
            return np.zeros(len(X))
        return self.tree.predict(X)
    
    def get_n_leaves(self):
        """Return number of leaves in the tree."""
        if self.tree is None:
            return 0
        try:
            if hasattr(self.tree, 'n_features_in_'):
                # Count leaves by traversing the tree structure
                summary = self.tree.summary()
                return len([k for k in summary.keys() if summary[k].get('children') == (None, None)])
        except Exception:
            pass
        return 1
    
    def is_single_leaf(self):
        """Check if tree is just a single leaf (no splits)."""
        return self.best_depth == 0 or self.get_n_leaves() <= 1
    
    def get_stats(self):
        """Return statistics about the M5 model."""
        return {
            'best_depth': self.best_depth,
            'n_leaves': self.get_n_leaves(),
            'cv_scores': self.cv_scores,
            'is_single_leaf': self.is_single_leaf()
        }
    
    def print_tree_rules(self):
        """
        Print the decision rules and linear models learned by the M5 tree.
        """
        feature_names = ['x', 'y', 'v_x', 'v_y']
        
        leaf_desc = M5_LEAF_REG_LABELS.get(self.m5_leaf_reg, self.m5_leaf_reg)
        print(f"\n  {'='*70}")
        print(f"  M5 MODEL TREE RULES for {self.var_name}_after ({leaf_desc}, depth={self.best_depth})")
        print(f"  Each leaf contains a LINEAR MODEL (not a constant!)")
        print(f"  {'='*70}")
        
        if self.tree is None:
            print(f"  No tree fitted yet")
            print(f"  {'='*70}")
            return
        
        # Depth 0: single global leaf model (no splits)
        if self.best_depth == 0 or isinstance(self.tree, (LinearRegression, Lasso, Ridge, ElasticNet)):
            single_label = {
                'none': 'Single OLS model (no splits)',
                'l1': 'Single Lasso (L1) model (no splits)',
                'l2': 'Single Ridge (L2) model (no splits)',
                'elasticnet': 'Single ElasticNet (L1+L2) model (no splits)',
            }.get(self.m5_leaf_reg, 'Single linear model (no splits)')
            print(f"\n  {single_label}:")
            coef = self.tree.coef_ if hasattr(self.tree, 'coef_') else [0,0,0,0]
            intercept = self.tree.intercept_ if hasattr(self.tree, 'intercept_') else 0
            self._print_linear_equation(intercept, coef, feature_names)
            print(f"  {'='*70}")
            return
        
        try:
            summary = self.tree.summary()
            
            # Print tree structure
            print(f"\n  Tree Structure:")
            self._print_tree_recursive(summary, 0, [], feature_names)
            
        except Exception as e:
            print(f"  Could not extract tree structure: {e}")
            
            # Fallback: show predictions for sample points
            print(f"\n  Sample predictions (showing learned behavior):")
            test_cases = [
                [400, 5, 100, -50],   # shallow impact, low speed
                [400, 5, 150, -100],  # medium impact
                [400, 5, 200, -150],  # steep impact, high speed
            ]
            for x, y, vx, vy in test_cases:
                pred = self.predict(np.array([[x, y, vx, vy]]))[0]
                print(f"    v_x={vx}, v_y={vy} → {self.var_name}_after = {pred:.2f}")
        
        print(f"  {'='*70}")
    
    def _print_tree_recursive(self, summary, node_id, conditions, feature_names):
        """Recursively print the tree structure with linear models at leaves."""
        if node_id not in summary:
            return
        
        node = summary[node_id]
        indent = "    " * (len(conditions) + 1)
        
        # Check if leaf
        children = node.get('children', (None, None))
        if children == (None, None) or children[0] is None:
            # Leaf node - print conditions and linear model
            print(f"\n{indent}Leaf {node_id}:")
            if conditions:
                cond_str = " AND ".join(conditions)
                print(f"{indent}  IF {cond_str}")
            else:
                print(f"{indent}  (root leaf - no conditions)")
            
            # Get the linear model for this leaf
            models = node.get('models', None)
            if models is not None:
                # models should contain the linear regression
                if hasattr(models, 'coef_'):
                    print(f"{indent}  THEN ", end="")
                    self._print_linear_equation(models.intercept_, models.coef_, feature_names, inline=True)
            else:
                # Try to get coefficients another way
                col = node.get('col', None)
                th = node.get('th', None)
                print(f"{indent}  THEN {self.var_name}_after = <linear model>")
            return
        
        # Internal node - get split info
        col = node.get('col', None)
        th = node.get('th', None)
        
        if col is not None and th is not None:
            feat_name = feature_names[col] if col < len(feature_names) else f"f{col}"
            
            # Left child (< threshold)
            left_cond = f"{feat_name} < {th:.2f}"
            self._print_tree_recursive(summary, children[0], conditions + [left_cond], feature_names)
            
            # Right child (>= threshold)
            right_cond = f"{feat_name} >= {th:.2f}"
            self._print_tree_recursive(summary, children[1], conditions + [right_cond], feature_names)
    
    def _print_linear_equation(self, intercept, coef, feature_names, inline=False):
        """Print a linear equation in readable format."""
        terms = []
        if abs(intercept) > 1e-6:
            terms.append(f"{intercept:.4f}")
        
        for c, name in zip(coef, feature_names):
            if abs(c) > 1e-6:
                if c > 0 and terms:
                    terms.append(f"+ {c:.4f}*{name}")
                elif c > 0:
                    terms.append(f"{c:.4f}*{name}")
                else:
                    terms.append(f"- {abs(c):.4f}*{name}")
        
        equation = " ".join(terms) if terms else "0"
        
        if inline:
            print(f"{self.var_name}_after = {equation}")
        else:
            print(f"    {self.var_name}_after = {equation}")


# Column order matches make_feature_vector: x, y, v_x, v_y
_PDDL_FLUENT_BY_COL = ("x_bird", "y_bird", "vx_bird", "vy_bird")


def _m5_pddl_split_atoms(col: int, th: float, bird: str) -> tuple[str, str]:
    """Left branch: feature < th; right branch: feature >= th (matches _print_tree_recursive)."""
    if col < 0 or col >= len(_PDDL_FLUENT_BY_COL):
        col = 0
    feat = _PDDL_FLUENT_BY_COL[col]
    ths = f"{float(th):.8f}"
    return (
        f"(< ({feat} {bird}) {ths})",
        f"(>= ({feat} {bird}) {ths})",
    )


def _m5_leaf_coef_intercept(models):
    if models is None or not hasattr(models, "coef_"):
        return None
    coef = np.asarray(models.coef_, dtype=float).reshape(-1)
    if coef.size < 4:
        pad = np.zeros(4, dtype=float)
        pad[: coef.size] = coef
        coef = pad
    else:
        coef = coef[:4].copy()
    intercept = float(getattr(models, "intercept_", 0.0))
    return coef, intercept


def m5_collision_leaves_for_pddl(m5: M5ModelTree, bird_param: str = "?b", debug: bool = False, var_label: str = ""):
    """
    Serialize an M5ModelTree into leaf paths and affine models for PDDL collision injection.

    Returns:
        List of (path_conditions, coef[4], intercept). Path is a list of PDDL comparison atoms.
        Empty list means the caller should fall back to the General (single affine) model.

    Column order for coef: x, y, v_x, v_y — same as make_feature_vector / inject_domain_file.
    """
    label = f"[{var_label}] " if var_label else ""

    if m5 is None or not isinstance(m5, M5ModelTree) or m5.tree is None:
        if debug:
            print(f"[M5-PDDL] {label}skip: no M5 model or tree (m5={type(m5).__name__ if m5 is not None else None})")
        return []

    bird = bird_param.strip()
    if not bird.startswith("?"):
        bird = "?" + bird.lstrip("?")

    if m5.best_depth == 0 or isinstance(m5.tree, (LinearRegression, Lasso, Ridge, ElasticNet)):
        pack = _m5_leaf_coef_intercept(m5.tree)
        if pack is None:
            if debug:
                print(f"[M5-PDDL] {label}depth-0 leaf: coef/intercept missing")
            return []
        if debug:
            print(
                f"[M5-PDDL] {label}depth=0 single affine (best_depth={m5.best_depth}, "
                f"leaf_reg={getattr(m5, 'm5_leaf_reg', '?')})"
            )
        return [([], pack[0], pack[1])]

    if not hasattr(m5.tree, "summary"):
        if debug:
            print(f"[M5-PDDL] {label}skip: tree has no summary() ({type(m5.tree).__name__})")
        return []

    try:
        summary = m5.tree.summary()
    except Exception as ex:
        if debug:
            print(f"[M5-PDDL] {label}summary() failed: {ex!r}")
        return []

    leaves: list[tuple[list[str], np.ndarray, float]] = []
    ok = {"v": True}

    def recurse(node_id: int, path: list[str]) -> None:
        if not ok["v"]:
            return
        if node_id not in summary:
            ok["v"] = False
            return
        node = summary[node_id]
        children = node.get("children", (None, None))
        if isinstance(children, list) and len(children) >= 2:
            children = (children[0], children[1])

        if children == (None, None) or children[0] is None:
            models = node.get("models")
            pack = _m5_leaf_coef_intercept(models)
            if pack is None:
                ok["v"] = False
                return
            leaves.append((list(path), pack[0], pack[1]))
            return

        col = node.get("col")
        th = node.get("th")
        if col is None or th is None:
            ok["v"] = False
            return
        left_atom, right_atom = _m5_pddl_split_atoms(int(col), float(th), bird)
        recurse(children[0], path + [left_atom])
        recurse(children[1], path + [right_atom])

    recurse(0, [])
    if not ok["v"] or len(leaves) == 0:
        if debug:
            print(
                f"[M5-PDDL] {label}tree walk failed or no leaves "
                f"(ok={ok['v']}, n_leaves={len(leaves)}, best_depth={getattr(m5, 'best_depth', None)})"
            )
        return []
    if debug:
        for i, (path, coef, icept) in enumerate(leaves):
            path_s = " ".join(path) if path else "(no split — global leaf)"
            print(
                f"[M5-PDDL] {label}leaf {i}: path={path_s} | intercept={icept:.6f} "
                f"coef={np.array2string(coef, precision=4, suppress_small=True)}"
            )
        print(f"[M5-PDDL] {label}serialized {len(leaves)} leaf/leaves for PDDL")
    return leaves


class PhysicsRatioModel:
    """
    A simple physics-based model that learns velocity ratios for collisions.
    
    For bounce physics:
        v_y_after = restitution_coefficient * v_y_before
        v_x_after = friction_coefficient * v_x_before
    
    This is more physically correct and sample-efficient than general linear regression.
    """
    def __init__(self, var_name):
        self.var_name = var_name
        self.ratio = None
        self.ratios = []  # Store all observed ratios
        self.ratios_filtered = []  # Ratios after outlier removal
        self.intercept_ = 0.0  # For compatibility with existing code
        self.coef_ = np.array([0.0, 0.0, 0.0, 0.0])  # [x, y, v_x, v_y]
        self.n_outliers_removed = 0
    
    def fit(self, pre_values, post_values):
        """
        Fit the model by computing robust ratio estimate.
        
        Uses median and filters outliers for more stable learning.
        
        Parameters:
            pre_values: array of pre-collision values for this variable
            post_values: array of post-collision values for this variable
        """
        self.ratios = []
        
        # Minimum pre-collision velocity to include in learning
        # For v_x: need significant horizontal velocity to measure friction accurately
        # For v_y: already filtered in pddl_agent.py, but double-check here
        MIN_PRE_VELOCITY = 30  # pixels/second
        
        for pre, post in zip(pre_values, post_values):
            if abs(pre) > MIN_PRE_VELOCITY:
                self.ratios.append(post / pre)
        
        if len(self.ratios) == 0:
            self.ratio = 0.5 if self.var_name == 'v_x' else -0.3  # Default physics
            self.ratios_filtered = []
            return self
        
        # Filter outliers using IQR method (for v_x which has high variance)
        if self.var_name == 'v_x' and len(self.ratios) >= 4:
            q1 = np.percentile(self.ratios, 25)
            q3 = np.percentile(self.ratios, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Also filter physically impossible values (ratio < 0 or > 1.5 for friction)
            self.ratios_filtered = [r for r in self.ratios 
                                    if lower_bound <= r <= upper_bound 
                                    and 0 <= r <= 1.5]
            self.n_outliers_removed = len(self.ratios) - len(self.ratios_filtered)
        else:
            self.ratios_filtered = self.ratios
            self.n_outliers_removed = 0
        
        # Use median for robustness (less sensitive to remaining outliers)
        if len(self.ratios_filtered) > 0:
            self.ratio = np.median(self.ratios_filtered)
        elif len(self.ratios) > 0:
            self.ratio = np.median(self.ratios)  # Fall back to all ratios
        else:
            self.ratio = 0.5 if self.var_name == 'v_x' else -0.3
        
        # Set coef_ for compatibility: the coefficient on the corresponding velocity
        # v_x -> coef_[2], v_y -> coef_[3]
        if self.var_name == 'v_x':
            self.coef_ = np.array([0.0, 0.0, self.ratio, 0.0])
        elif self.var_name == 'v_y':
            self.coef_ = np.array([0.0, 0.0, 0.0, self.ratio])
        
        return self
    
    def predict(self, X):
        """
        Predict post-collision value using learned ratio.
        
        Parameters:
            X: array of shape (n_samples, 4) with [x, y, v_x, v_y]
        
        Returns:
            Predicted post-collision values
        """
        if self.ratio is None:
            return np.zeros(len(X))
        
        # X columns: [x, y, v_x, v_y] -> indices 0, 1, 2, 3
        if self.var_name == 'v_x':
            return self.ratio * X[:, 2]  # v_x is column 2
        elif self.var_name == 'v_y':
            return self.ratio * X[:, 3]  # v_y is column 3
        else:
            return np.zeros(len(X))
    
    def get_stats(self):
        """Return statistics about the learned ratio."""
        if len(self.ratios) == 0:
            return {
                'mean': 0, 'std': 0, 'median': 0,
                'n_samples': 0, 'n_filtered': 0, 'n_outliers': 0
            }
        
        filtered = self.ratios_filtered if len(self.ratios_filtered) > 0 else self.ratios
        return {
            'mean': np.mean(self.ratios),
            'std': np.std(self.ratios),
            'median': np.median(filtered),
            'n_samples': len(self.ratios),
            'n_filtered': len(self.ratios_filtered),
            'n_outliers': self.n_outliers_removed
        }


class AngleDependentFrictionModel:
    """
    Learns v_x friction as a function of impact angle.
    
    Physics intuition:
    - Steep impact (large |v_y/v_x|): More friction, lower v_x retention
    - Shallow impact (small |v_y/v_x|): Less friction, higher v_x retention
    
    Model: v_x_ratio = base_ratio + angle_coef * impact_angle_factor
    Where impact_angle_factor = |v_y| / (|v_x| + |v_y|)  (0 = horizontal, 1 = vertical)
    """
    def __init__(self):
        self.base_ratio = 0.7  # v_x retention for horizontal impact
        self.angle_coef = -0.5  # How much steeper angle reduces retention
        self.intercept_ = 0.0
        self.coef_ = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Store training data for analysis
        self.impact_factors = []
        self.observed_ratios = []
        self.linear_model = None
    
    def fit(self, pre_states, pre_vx_values, post_vx_values):
        """
        Fit the angle-dependent friction model.
        
        Parameters:
            pre_states: list of pre-collision state dicts with v_x and v_y
            pre_vx_values: array of pre-collision v_x values
            post_vx_values: array of post-collision v_x values
        """
        self.impact_factors = []
        self.observed_ratios = []
        
        MIN_VX = 30  # Minimum v_x to include
        
        for state, pre_vx, post_vx in zip(pre_states, pre_vx_values, post_vx_values):
            if abs(pre_vx) < MIN_VX:
                continue
            
            v_y = state['v_y']
            v_x = state['v_x']
            
            # Impact angle factor: 0 = horizontal impact, 1 = vertical impact
            # Using |v_y| / (|v_x| + |v_y|) as a normalized measure
            total_speed = abs(v_x) + abs(v_y)
            if total_speed > 0:
                impact_factor = abs(v_y) / total_speed
            else:
                impact_factor = 0.5
            
            ratio = post_vx / pre_vx
            
            # Filter physically impossible values only
            # ratio = 0 is valid for steep impacts where bird bounces in place
            # ratio > 1.5 means gained energy (impossible for friction)
            if 0 <= ratio <= 1.5:
                self.impact_factors.append(impact_factor)
                self.observed_ratios.append(ratio)
        
        if len(self.impact_factors) < 2:
            # Not enough data - use simple average
            if len(self.observed_ratios) > 0:
                self.base_ratio = np.median(self.observed_ratios)
            self.angle_coef = 0
            return self
        
        # Fit linear model: ratio = base + coef * impact_factor
        X = np.array(self.impact_factors).reshape(-1, 1)
        y = np.array(self.observed_ratios)
        
        self.linear_model = LinearRegression()
        self.linear_model.fit(X, y)
        
        self.base_ratio = self.linear_model.intercept_
        self.angle_coef = self.linear_model.coef_[0]
        
        # Clamp to reasonable physics (base should be 0.3-1.0, angle effect should be negative or small positive)
        self.base_ratio = np.clip(self.base_ratio, 0.3, 1.0)
        # angle_coef can be negative (steeper = more friction) or small positive
        
        return self
    
    def predict(self, X):
        """
        Predict v_x_after given pre-collision state.
        
        Parameters:
            X: array of shape (n_samples, 4) with [x, y, v_x, v_y]
        """
        predictions = []
        for row in X:
            v_x = row[2]
            v_y = row[3]
            
            # Calculate impact factor
            total_speed = abs(v_x) + abs(v_y)
            if total_speed > 0:
                impact_factor = abs(v_y) / total_speed
            else:
                impact_factor = 0.5
            
            # Predict ratio based on impact angle
            predicted_ratio = self.base_ratio + self.angle_coef * impact_factor
            predicted_ratio = np.clip(predicted_ratio, 0, 1.5)  # Physical bounds
            
            predictions.append(predicted_ratio * v_x)
        
        return np.array(predictions)
    
    def predict_ratio(self, v_x, v_y):
        """Predict the v_x retention ratio for given velocities."""
        total_speed = abs(v_x) + abs(v_y)
        if total_speed > 0:
            impact_factor = abs(v_y) / total_speed
        else:
            impact_factor = 0.5
        
        ratio = self.base_ratio + self.angle_coef * impact_factor
        return np.clip(ratio, 0, 1.5)
    
    def get_stats(self):
        """Return statistics about the learned model."""
        if len(self.observed_ratios) == 0:
            return {
                'base_ratio': self.base_ratio,
                'angle_coef': self.angle_coef,
                'n_samples': 0,
                'r_squared': 0
            }
        
        # Calculate R-squared if we have a linear model
        r_squared = 0
        if self.linear_model is not None and len(self.impact_factors) >= 2:
            X = np.array(self.impact_factors).reshape(-1, 1)
            y = np.array(self.observed_ratios)
            r_squared = self.linear_model.score(X, y)
        
        return {
            'base_ratio': self.base_ratio,
            'angle_coef': self.angle_coef,
            'n_samples': len(self.observed_ratios),
            'r_squared': r_squared,
            'mean_ratio': np.mean(self.observed_ratios),
            'std_ratio': np.std(self.observed_ratios)
        }


class EventModelManager:
    """
    Model comparison for event learning: General vs CART vs M5 (L1 + ElasticNet leaves only).
    
    Trains General (for PDDL injection), CART, and M5 model trees with leaf
    regularization: none (OLS), L1, L2, ElasticNet (alpha/l1_ratio from config).
    Compares all via LOO-CV; General is still always used for PDDL injection.
    """
    
    def __init__(self):
        self.comparison_results = {}  # Store results for debug output
    
    def train_and_compare(self, event_name, var_name, X, y, pre_states=None):
        """
        Train M5 trees for leaf reg: L1, ElasticNet (injected into PDDL).

        General and CART comparison training is disabled for speed — only M5
        is fitted and selected for injection.
        
        Returns:
            dict with m5_models (by reg key), m5_stats_by_reg,
            m5_model / m5_stats (best M5 by LOO-CV among trained leaf regs), etc.
        """
        n_samples = len(y)

        # --- General + CART (disabled — comparison-only, not injected) ---
        # config = REGULARIZATION_CONFIG
        # general_model = self._train_general_model(
        #     X, y,
        #     alpha=config['alpha'],
        #     l1_ratio=config['l1_ratio'],
        #     regularization=config['type']
        # )
        # general_stats = self._compute_model_stats(general_model, X, y, 'general')
        # cart_model = CARTEventModel(var_name)
        # cart_model.fit(X, y)
        # cart_stats = self._compute_model_stats(cart_model, X, y, 'cart')
        general_model = None
        general_stats = {'loo_cv': float('inf'), 'loo_cv_std': 0, 'train_rmse': float('inf'), 'r2': 0, 'n_samples': n_samples}
        cart_model = None
        cart_stats = {'loo_cv': float('inf'), 'loo_cv_std': 0, 'train_rmse': float('inf'), 'r2': 0, 'n_samples': n_samples}

        m5_models = {}
        m5_stats_by_reg = {}
        for reg in M5_LEAF_REG_ORDER:
            m = M5ModelTree(var_name, m5_leaf_reg=reg)
            m.fit(X, y)
            m5_models[reg] = m
            m5_stats_by_reg[reg] = self._compute_model_stats(m, X, y, 'm5')
        
        def _loo_key(r):
            v = m5_stats_by_reg[r]['loo_cv']
            return v if np.isfinite(v) else float('inf')
        
        best_m5_reg = min(M5_LEAF_REG_ORDER, key=_loo_key)
        m5_model = m5_models[best_m5_reg]
        m5_stats = m5_stats_by_reg[best_m5_reg]

        info = m5_model.get_stats()
        winner_name = f"{M5_LEAF_REG_LABELS[best_m5_reg]} (d={info['best_depth']}, l={info['n_leaves']})"
        winner = m5_model
        improvement_pct = 0.0
        
        result = {
            'winner': winner,
            'winner_name': winner_name,
            'general_model': general_model,
            'general_stats': general_stats,
            'cart_model': cart_model,
            'cart_stats': cart_stats,
            'm5_models': m5_models,
            'm5_stats_by_reg': m5_stats_by_reg,
            'm5_model': m5_model,
            'm5_stats': m5_stats,
            'best_m5_leaf_reg': best_m5_reg,
            'domain_model': m5_model,
            'domain_stats': m5_stats,
            'domain_name': 'M5',
            'improvement_pct': improvement_pct,
            'n_samples': n_samples
        }
        
        self.comparison_results[var_name] = result
        
        return result
    
    def train_ablation_comparison(self, var_name, X_base, X_extended, y):
        """
        Compare model performance with and without the velocity ratio feature.
        
        This is an A/B test to validate whether adding velocity_ratio improves
        collision model learning performance.
        
        Parameters:
            var_name: Name of target variable ('v_x', 'v_y', 'y')
            X_base: Feature matrix WITHOUT velocity_ratio [x, y, v_x, v_y] shape (n, 4)
            X_extended: Feature matrix WITH velocity_ratio [x, y, v_x, v_y, ratio] shape (n, 5)
            y: Target values (post-collision values)
        
        Returns:
            dict with:
            - baseline_stats: LOO-CV, train_rmse, r2 for base features
            - extended_stats: LOO-CV, train_rmse, r2 for extended features
            - improvement_pct: % improvement in LOO-CV (positive = extended is better)
            - feature_helps: bool - True if extended LOO-CV is >5% better
            - feature_hurts: bool - True if extended LOO-CV is worse (overfitting)
        """
        n_samples = len(y)
        
        if n_samples < 3:
            return {
                'baseline_stats': {'loo_cv': float('inf'), 'train_rmse': float('inf'), 'r2': 0},
                'extended_stats': {'loo_cv': float('inf'), 'train_rmse': float('inf'), 'r2': 0},
                'improvement_pct': 0,
                'feature_helps': False,
                'feature_hurts': False,
                'n_samples': n_samples
            }
        
        config = REGULARIZATION_CONFIG
        
        # Train baseline model (4 features: x, y, v_x, v_y)
        baseline_model = self._train_general_model(
            X_base, y,
            alpha=config['alpha'],
            l1_ratio=config['l1_ratio'],
            regularization=config['type']
        )
        baseline_stats = self._compute_model_stats(baseline_model, X_base, y, 'general')
        
        # Train extended model (5 features: x, y, v_x, v_y, velocity_ratio)
        extended_model = self._train_general_model(
            X_extended, y,
            alpha=config['alpha'],
            l1_ratio=config['l1_ratio'],
            regularization=config['type']
        )
        extended_stats = self._compute_model_stats(extended_model, X_extended, y, 'general')
        
        # Calculate improvement
        baseline_loo = baseline_stats['loo_cv']
        extended_loo = extended_stats['loo_cv']
        
        if baseline_loo > 0 and np.isfinite(baseline_loo) and np.isfinite(extended_loo):
            improvement_pct = ((baseline_loo - extended_loo) / baseline_loo) * 100
        else:
            improvement_pct = 0
        
        # Decision criteria
        SIGNIFICANCE_THRESHOLD = 5.0  # 5% improvement threshold
        feature_helps = improvement_pct > SIGNIFICANCE_THRESHOLD
        feature_hurts = improvement_pct < -SIGNIFICANCE_THRESHOLD
        
        return {
            'baseline_model': baseline_model,
            'baseline_stats': baseline_stats,
            'extended_model': extended_model,
            'extended_stats': extended_stats,
            'improvement_pct': improvement_pct,
            'feature_helps': feature_helps,
            'feature_hurts': feature_hurts,
            'n_samples': n_samples
        }
    
    def train_multi_feature_comparison(self, var_name, feature_sets, y):
        """
        Compare model performance across multiple feature sets.
        
        This compares different combinations of features to find the best set
        for predicting post-collision state variables.
        
        Parameters:
            var_name: Name of target variable ('v_x', 'v_y', 'y')
            feature_sets: dict mapping feature set name to feature matrix
                e.g., {'base': X_base, 'ratio': X_ratio, 'trig': X_trig, 'all': X_all}
            y: Target values (post-collision values)
        
        Returns:
            dict with results for each feature set:
            - stats: LOO-CV, train_rmse, r2
            - model: trained model
            - n_features: number of features
        """
        n_samples = len(y)
        results = {}
        
        if n_samples < 3:
            for idx, name in enumerate(feature_sets):
                results[name] = {
                    'stats': {'loo_cv': float('inf'), 'train_rmse': float('inf'), 'r2': 0},
                    'model': None,
                    'n_features': feature_sets[name].shape[1] if len(feature_sets[name]) > 0 else 0,
                    'n_samples': n_samples,
                    'improvement_vs_base': 0,
                    'is_best': (idx == 0)  # First one is "best" by default when no data
                }
            return results
        
        config = REGULARIZATION_CONFIG
        
        for name, X in feature_sets.items():
            model = self._train_general_model(
                X, y,
                alpha=config['alpha'],
                l1_ratio=config['l1_ratio'],
                regularization=config['type']
            )
            stats = self._compute_model_stats(model, X, y, 'general')
            
            results[name] = {
                'stats': stats,
                'model': model,
                'n_features': X.shape[1],
                'n_samples': n_samples
            }
        
        # Find best feature set (lowest LOO-CV)
        best_name = min(results.keys(), key=lambda k: results[k]['stats']['loo_cv'])
        baseline_loo = results.get('base', results[list(results.keys())[0]])['stats']['loo_cv']
        
        # Add comparison info to each result
        for name, result in results.items():
            loo = result['stats']['loo_cv']
            if baseline_loo > 0 and np.isfinite(baseline_loo) and np.isfinite(loo):
                improvement = ((baseline_loo - loo) / baseline_loo) * 100
            else:
                improvement = 0
            result['improvement_vs_base'] = improvement
            result['is_best'] = (name == best_name)
        
        return results
    
    def _train_general_model(self, X, y, alpha=1.0, l1_ratio=0.5, regularization='elasticnet'):
        """
        Train a general regression model with configurable regularization.
        
        Regularization Options:
        ─────────────────────────────────────────────────────────────────────────
        1. 'none' (OLS):      L = Σ(y - ŷ)²
                              No penalty, prone to overfitting
        
        2. 'l1' (Lasso):      L = Σ(y - ŷ)² + α·Σ|βⱼ|
                              Drives coefficients to EXACTLY ZERO (sparse)
        
        3. 'l2' (Ridge):      L = Σ(y - ŷ)² + α·Σβⱼ²
                              Shrinks ALL coefficients (stable)
        
        4. 'elasticnet':      L = Σ(y - ŷ)² + α·[ρ·Σ|βⱼ| + (1-ρ)/2·Σβⱼ²]
                              Combines L1 sparsity + L2 stability
        ─────────────────────────────────────────────────────────────────────────
        
        Parameters:
            X: Feature matrix [x, y, v_x, v_y]
            y: Target values (post-collision values)
            alpha: Regularization strength (higher = more regularization)
            l1_ratio: For ElasticNet, balance between L1 and L2 (0=Ridge, 1=Lasso)
            regularization: 'none', 'l1', 'l2', or 'elasticnet'
        
        Returns:
            Trained model with poly_features and model_type attributes
        """
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        if regularization == 'none':
            # Version 1: No Regularization (Ordinary Least Squares)
            # L = Σ(y - ŷ)²
            model = LinearRegression()
            model.fit(X_poly, y)
            model.model_type = 'general_ols'
            model.alpha = 0
            model.l1_ratio = None
            
        elif regularization == 'l1':
            # Version 2: L1 Regularization (Lasso)
            # L = Σ(y - ŷ)² + α·Σ|βⱼ|
            # → Drives irrelevant coefficients to EXACTLY ZERO
            model = Lasso(alpha=alpha, max_iter=10000)
            model.fit(X_poly, y)
            model.model_type = 'general_lasso'
            model.alpha = alpha
            model.l1_ratio = 1.0
            
        elif regularization == 'l2':
            # Version 3: L2 Regularization (Ridge)
            # L = Σ(y - ŷ)² + α·Σβⱼ²
            # → Shrinks ALL coefficients toward zero
            model = Ridge(alpha=alpha, max_iter=10000)
            model.fit(X_poly, y)
            model.model_type = 'general_ridge'
            model.alpha = alpha
            model.l1_ratio = 0.0
            
        else:  # elasticnet (default)
            # Version 4: L1 + L2 Regularization (ElasticNet)
            # L = Σ(y - ŷ)² + α·[ρ·Σ|βⱼ| + (1-ρ)/2·Σβⱼ²]
            # → Combines sparsity (L1) with stability (L2)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
            model.fit(X_poly, y)
            model.model_type = 'general_elasticnet'
            model.alpha = alpha
            model.l1_ratio = l1_ratio
        
        # Store poly transformer for predictions
        model.poly_features = poly
        
        return model
    
    def _compute_model_stats(self, model, X, y, model_type):
        """Compute training RMSE, LOO-CV RMSE, and R² for a model."""
        n_samples = len(y)
        
        # Get predictions
        if model_type == 'general':
            X_poly = model.poly_features.transform(X)
            predictions = model.predict(X_poly)
        else:
            # CART or M5 model
            predictions = model.predict(X)
        
        # Training RMSE
        train_mse = np.mean((predictions - y) ** 2)
        train_rmse = np.sqrt(train_mse)
        
        # R² score
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # LOO-CV RMSE and standard error
        loo_cv, loo_cv_std = self._compute_loo_cv(model, X, y, model_type)
        
        return {
            'train_rmse': train_rmse,
            'loo_cv': loo_cv,
            'loo_cv_std': loo_cv_std,  # Standard error for variance bands
            'r2': r2,
            'n_samples': n_samples
        }
    
    def _compute_loo_cv(self, model, X, y, model_type):
        """Compute Leave-One-Out Cross-Validation RMSE and standard error."""
        n_samples = len(y)
        
        if n_samples < 2:
            return float('inf'), float('inf')
        
        loo_errors = []
        
        for i in range(n_samples):
            # Create train/test split leaving out sample i
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            X_test = X[i:i+1]
            y_test = y[i]
            
            try:
                if model_type == 'general':
                    # Retrain general model with same regularization type
                    # MUST use degree=2 to match _train_general_model()
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    X_train_poly = poly.fit_transform(X_train)
                    X_test_poly = poly.transform(X_test)
                    
                    # Get regularization parameters from original model
                    alpha = getattr(model, 'alpha', 1.0)
                    l1_ratio = getattr(model, 'l1_ratio', 0.5)
                    model_type_str = getattr(model, 'model_type', 'general_elasticnet')
                    
                    # Create appropriate model based on regularization type
                    if model_type_str == 'general_ols':
                        temp_model = LinearRegression()
                    elif model_type_str == 'general_lasso':
                        temp_model = Lasso(alpha=alpha, max_iter=10000)
                    elif model_type_str == 'general_ridge':
                        temp_model = Ridge(alpha=alpha, max_iter=10000)
                    else:
                        temp_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
                    
                    temp_model.fit(X_train_poly, y_train)
                    pred = temp_model.predict(X_test_poly)[0]
                    
                elif model_type == 'cart':
                    # Retrain CART model with same configuration
                    best_depth = model.best_depth if model.best_depth else 2
                    min_leaf = max(1, min(3, len(y_train) // (2 ** best_depth + 1)))
                    
                    temp_tree = DecisionTreeRegressor(
                        max_depth=best_depth,
                        min_samples_leaf=min_leaf,
                        random_state=42
                    )
                    temp_tree.fit(X_train, y_train)
                    pred = temp_tree.predict(X_test)[0]
                    
                elif model_type == 'm5':
                    leaf_reg = getattr(model, 'm5_leaf_reg', 'l1')
                    best_depth = model.best_depth if model.best_depth else 0
                    
                    if best_depth == 0 or len(y_train) < 4:
                        temp_m5 = _m5_leaf_estimator(leaf_reg)
                        temp_m5.fit(X_train, y_train)
                    else:
                        try:
                            temp_m5 = LinearTreeRegressor(
                                base_estimator=_m5_leaf_estimator(leaf_reg),
                                max_depth=min(best_depth, M5ModelTree.MAX_DEPTH_CAP),
                                min_samples_leaf=3  # LinearTreeRegressor requires > 2
                            )
                            temp_m5.fit(X_train, y_train)
                        except Exception:
                            temp_m5 = _m5_leaf_estimator(leaf_reg)
                            temp_m5.fit(X_train, y_train)
                    pred = temp_m5.predict(X_test)[0]
                else:
                    # Fallback for any other model type
                    pred = model.predict(X_test)[0]
                
                loo_errors.append((pred - y_test) ** 2)
            except Exception:
                # If LOO fails for this sample, skip it
                continue
        
        if len(loo_errors) == 0:
            return float('inf'), float('inf')
        
        rmse = np.sqrt(np.mean(loo_errors))
        # Standard error of LOO-CV (variance of individual errors)
        std_error = np.std(np.sqrt(loo_errors)) if len(loo_errors) > 1 else 0
        
        return rmse, std_error
    
    def _select_winner(self, general_model, general_stats, cart_model, cart_stats):
        """Select the best model based on LOO-CV RMSE (legacy 2-way comparison)."""
        # Get the general model type name for display
        model_type_str = getattr(general_model, 'model_type', 'general_elasticnet')
        model_type_display = {
            'general_ols': 'General (OLS)',
            'general_lasso': 'General (Lasso/L1)',
            'general_ridge': 'General (Ridge/L2)',
            'general_elasticnet': 'General (ElasticNet/L1+L2)'
        }.get(model_type_str, 'General')
        
        general_loo = general_stats['loo_cv']
        cart_loo = cart_stats['loo_cv']
        
        # Handle infinite or invalid LOO-CV values
        if not np.isfinite(general_loo):
            general_loo = float('inf')
        if not np.isfinite(cart_loo):
            cart_loo = float('inf')
        
        # Generate CART display name with tree info
        cart_stats_info = cart_model.get_stats()
        cart_display = f"CART (depth={cart_stats_info['best_depth']}, leaves={cart_stats_info['n_leaves']})"
        
        if cart_loo < general_loo:
            # CART wins
            if general_loo > 0 and np.isfinite(general_loo):
                improvement = ((general_loo - cart_loo) / general_loo) * 100
            else:
                improvement = 0
            return cart_model, cart_display, improvement
        else:
            # General model wins
            if cart_loo > 0 and np.isfinite(cart_loo):
                improvement = ((cart_loo - general_loo) / cart_loo) * 100
            else:
                improvement = 0
            return general_model, model_type_display, improvement
    
    def _select_winner_m5_suite(self, general_model, general_stats, cart_model, cart_stats,
                                m5_models, m5_stats_by_reg):
        """Select best model by LOO-CV among General, CART, and all M5 leaf-regularization variants."""
        model_type_str = getattr(general_model, 'model_type', 'general_elasticnet')
        model_type_display = {
            'general_ols': 'General (OLS)',
            'general_lasso': 'General (Lasso/L1)',
            'general_ridge': 'General (Ridge/L2)',
            'general_elasticnet': 'General (ElasticNet/L1+L2)'
        }.get(model_type_str, 'General')
        
        general_loo = general_stats['loo_cv'] if np.isfinite(general_stats['loo_cv']) else float('inf')
        cart_loo = cart_stats['loo_cv'] if np.isfinite(cart_stats['loo_cv']) else float('inf')
        
        cart_info = cart_model.get_stats()
        cart_display = f"CART (d={cart_info['best_depth']}, l={cart_info['n_leaves']})"
        
        candidates = [
            (general_model, model_type_display, general_loo),
            (cart_model, cart_display, cart_loo),
        ]
        for reg in M5_LEAF_REG_ORDER:
            m = m5_models[reg]
            st = m5_stats_by_reg[reg]
            lo = st['loo_cv'] if np.isfinite(st['loo_cv']) else float('inf')
            info = m.get_stats()
            name = f"{M5_LEAF_REG_LABELS[reg]} (d={info['best_depth']}, l={info['n_leaves']})"
            candidates.append((m, name, lo))
        
        best_model, best_name, best_loo = min(candidates, key=lambda x: x[2])
        
        if general_loo > 0 and np.isfinite(general_loo) and best_loo < general_loo:
            improvement = ((general_loo - best_loo) / general_loo) * 100
        else:
            improvement = 0
        
        return best_model, best_name, improvement
    
    def get_debug_output(self, n_samples=None):
        """Generate formatted debug output for all compared models."""
        lines = []
        
        for var_name, result in self.comparison_results.items():
            n = result['n_samples'] if n_samples is None else n_samples
            
            # Get the general model type for display
            general_model = result['general_model']
            model_type_str = getattr(general_model, 'model_type', 'general_elasticnet')
            model_type_display = {
                'general_ols': 'General (OLS)',
                'general_lasso': 'General (Lasso/L1)',
                'general_ridge': 'General (Ridge/L2)',
                'general_elasticnet': 'General (ElasticNet)'
            }.get(model_type_str, 'General')
            
            lines.append("")
            lines.append("=" * 70)
            lines.append(f"MODEL COMPARISON: {var_name} ({n} samples) — General, CART, M5×{len(M5_LEAF_REG_ORDER)}")
            lines.append("=" * 70)
            lines.append(f"{'Model':<42} {'Train RMSE':<12} {'LOO-CV':<12} {'R²':<10}")
            lines.append("-" * 70)
            
            # General model row
            gen = result['general_stats']
            general_marker = " [INJECTED]" if result['winner_name'].startswith('General') else ""
            lines.append(f"{(model_type_display + general_marker):<42} {gen['train_rmse']:<12.4f} {gen['loo_cv']:<12.4f} {gen['r2']:<10.4f}")
            
            # CART model row
            cart = result['cart_stats']
            cart_model = result['cart_model']
            cart_info = cart_model.get_stats()
            cart_label = f"CART (d={cart_info['best_depth']}, l={cart_info['n_leaves']})"
            lines.append(f"{cart_label:<42} {cart['train_rmse']:<12.4f} {cart['loo_cv']:<12.4f} {cart['r2']:<10.4f}")
            
            # M5 variants (leaf regularization)
            if result.get('m5_stats_by_reg') and result.get('m5_models'):
                for reg in M5_LEAF_REG_ORDER:
                    m5 = result['m5_stats_by_reg'][reg]
                    m5_model = result['m5_models'][reg]
                    m5_info = m5_model.get_stats()
                    short = M5_LEAF_REG_LABELS[reg]
                    m5_label = f"{short} (d={m5_info['best_depth']}, l={m5_info['n_leaves']})"
                    lines.append(f"{m5_label:<42} {m5['train_rmse']:<12.4f} {m5['loo_cv']:<12.4f} {m5['r2']:<10.4f}")
            elif 'm5_stats' in result and 'm5_model' in result:
                m5 = result['m5_stats']
                m5_model = result['m5_model']
                m5_info = m5_model.get_stats()
                m5_label = f"M5 (d={m5_info['best_depth']}, l={m5_info['n_leaves']})"
                lines.append(f"{m5_label:<42} {m5['train_rmse']:<12.4f} {m5['loo_cv']:<12.4f} {m5['r2']:<10.4f}")
            
            # Winner line
            lines.append("-" * 70)
            winner_str = f"BEST MODEL: {result['winner_name']}"
            if result['improvement_pct'] > 0:
                winner_str += f" ({result['improvement_pct']:.1f}% better LOO-CV vs General)"
            lines.append(winner_str)
            lines.append("NOTE: General model is ALWAYS used for PDDL injection")
            
            # Details of selected model
            lines.extend(self._format_model_coefficients(var_name, result['winner']))
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def _format_model_coefficients(self, var_name, model):
        """Format the coefficients/rules of a model for display."""
        lines = []
        
        if isinstance(model, CARTEventModel):
            # Format CART tree rules
            stats = model.get_stats()
            if stats['is_single_leaf']:
                # Single leaf = just a constant (mean prediction)
                leaf_value = model.tree.tree_.value[0, 0, 0]
                lines.append(f"\n  {var_name}_after = {leaf_value:.4f} (constant)")
            else:
                lines.append(f"\n  CART Decision Rules:")
                rules = model.get_tree_rules()
                feature_names = ['x', 'y', 'v_x', 'v_y']
                for conditions, leaf_value in rules:
                    if conditions:
                        cond_strs = []
                        for feat_idx, threshold, direction in conditions:
                            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"f{feat_idx}"
                            cond_strs.append(f"{feat_name} {direction} {threshold:.2f}")
                        lines.append(f"    IF {' AND '.join(cond_strs)}:")
                        lines.append(f"       → {var_name}_after = {leaf_value:.4f}")
                    else:
                        lines.append(f"    DEFAULT: {var_name}_after = {leaf_value:.4f}")
            
            # Show CV scores for different depths
            if stats['cv_scores']:
                lines.append(f"  └─ CV scores by depth: {stats['cv_scores']}")
        
        elif isinstance(model, M5ModelTree):
            reg_lbl = M5_LEAF_REG_LABELS.get(model.m5_leaf_reg, model.m5_leaf_reg)
            lines.append(f"\n  M5 Model Tree ({reg_lbl}):")
            stats = model.get_stats()
            if stats['is_single_leaf']:
                lines.append(f"    (Single leaf - global linear model)")
            else:
                lines.append(f"    Depth: {stats['best_depth']}, Leaves: {stats['n_leaves']}")
            lines.append(f"    See console output above for detailed tree structure.")
                
        elif hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            # General linear model
            coef_names = ["x", "y", "v_x", "v_y"]
            terms = [f"{model.intercept_:.4f}"]
            for coef, name in zip(model.coef_, coef_names):
                if abs(coef) > 0.0001:
                    terms.append(f"({coef:.4f})*{name}")
            lines.append(f"\n  {var_name}_after = " + " + ".join(terms))
        
        return lines


def print_all_regularization_equations(X, y, var_name, alpha=1.0, l1_ratio=0.5):
    """
    Train and print all 4 regularization versions for a variable.
    
    Parameters:
        X: Feature matrix [x, y, v_x, v_y]
        y: Target values (post-collision values)
        var_name: Variable name ('v_x' or 'v_y')
        alpha: Regularization strength
        l1_ratio: For ElasticNet, balance between L1 and L2
    """
    poly = PolynomialFeatures(degree=1, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    coef_names = ["x", "y", "v_x", "v_y"]
    
    print()
    print("=" * 80)
    print(f"  REGULARIZATION COMPARISON FOR {var_name.upper()}")
    print("=" * 80)
    
    # Version 1: No Regularization (OLS)
    model_ols = LinearRegression()
    model_ols.fit(X_poly, y)
    eq_ols = _format_equation(var_name, model_ols.intercept_, model_ols.coef_, coef_names)
    print()
    print("  1. WITHOUT REGULARIZATION (OLS)")
    print("     ─────────────────────────────────────────────────────────────────")
    print(f"     Loss: L = Σ(y - ŷ)²")
    print(f"     {eq_ols}")
    
    # Version 2: L1 Regularization (Lasso)
    model_l1 = Lasso(alpha=alpha, max_iter=10000)
    model_l1.fit(X_poly, y)
    eq_l1 = _format_equation(var_name, model_l1.intercept_, model_l1.coef_, coef_names)
    print()
    print(f"  2. WITH L1 REGULARIZATION (Lasso, α={alpha})")
    print("     ─────────────────────────────────────────────────────────────────")
    print(f"     Loss: L = Σ(y - ŷ)² + α·Σ|βⱼ|")
    print(f"     {eq_l1}")
    
    # Version 3: L2 Regularization (Ridge)
    model_l2 = Ridge(alpha=alpha, max_iter=10000)
    model_l2.fit(X_poly, y)
    eq_l2 = _format_equation(var_name, model_l2.intercept_, model_l2.coef_, coef_names)
    print()
    print(f"  3. WITH L2 REGULARIZATION (Ridge, α={alpha})")
    print("     ─────────────────────────────────────────────────────────────────")
    print(f"     Loss: L = Σ(y - ŷ)² + α·Σβⱼ²")
    print(f"     {eq_l2}")
    
    # Version 4: L1 + L2 Regularization (ElasticNet)
    model_en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    model_en.fit(X_poly, y)
    eq_en = _format_equation(var_name, model_en.intercept_, model_en.coef_, coef_names)
    print()
    print(f"  4. WITH L1+L2 REGULARIZATION (ElasticNet, α={alpha}, ρ={l1_ratio})")
    print("     ─────────────────────────────────────────────────────────────────")
    print(f"     Loss: L = Σ(y - ŷ)² + α·[ρ·Σ|βⱼ| + (1-ρ)/2·Σβⱼ²]")
    print(f"     {eq_en}")
    
    print()
    print("=" * 80)


def _format_equation(var_name, intercept, coefs, coef_names):
    """Format coefficients as a readable equation string."""
    terms = []
    
    # Add intercept if significant
    if abs(intercept) > 1e-6:
        terms.append(f"{intercept:.4f}")
    
    # Add coefficient terms
    for coef, name in zip(coefs, coef_names):
        if abs(coef) > 1e-6:
            if coef >= 0:
                terms.append(f"+ {coef:.4f}·{name}")
            else:
                terms.append(f"- {abs(coef):.4f}·{name}")
    
    if not terms:
        return f"{var_name}_after = 0"
    
    # Clean up the first term (remove leading +)
    eq = " ".join(terms)
    if eq.startswith("+ "):
        eq = eq[2:]
    
    return f"{var_name}_after = {eq}"


# Default regularization configuration
# Change this to switch between different regularization methods
REGULARIZATION_CONFIG = {
    'type': 'elasticnet',  # Options: 'none', 'l1', 'l2', 'elasticnet' — controls General model only
    'alpha': 10.0,         # Strength for General; also used for each trained M5 leaf estimator
    'l1_ratio': 0.5        # For General ElasticNet and for M5 ElasticNet leaves
}


def set_regularization(regularization_type='elasticnet', alpha=1.0, l1_ratio=0.5):
    """
    Configure the regularization method for event learning.
    
    Regularization Types:
    ─────────────────────────────────────────────────────────────────────────
    'none':      L = Σ(y - ŷ)²
                 No penalty, prone to overfitting
    
    'l1':        L = Σ(y - ŷ)² + α·Σ|βⱼ|
                 Drives coefficients to EXACTLY ZERO (sparse)
    
    'l2':        L = Σ(y - ŷ)² + α·Σβⱼ²
                 Shrinks ALL coefficients (stable)
    
    'elasticnet': L = Σ(y - ŷ)² + α·[ρ·Σ|βⱼ| + (1-ρ)/2·Σβⱼ²]
                  Combines L1 sparsity + L2 stability
    ─────────────────────────────────────────────────────────────────────────
    
    Parameters:
        regularization_type: 'none', 'l1', 'l2', or 'elasticnet'
        alpha: Regularization strength (ignored for 'none')
        l1_ratio: For ElasticNet, balance between L1 and L2
    
    Example:
        >>> set_regularization('l1', alpha=0.5)  # Lasso with moderate penalty
        >>> set_regularization('l2', alpha=1.0)  # Ridge with standard penalty
        >>> set_regularization('elasticnet', alpha=1.0, l1_ratio=0.7)  # More L1
    """
    global REGULARIZATION_CONFIG
    REGULARIZATION_CONFIG['type'] = regularization_type
    REGULARIZATION_CONFIG['alpha'] = alpha
    REGULARIZATION_CONFIG['l1_ratio'] = l1_ratio
    
    print(f"Regularization set to: {regularization_type}")
    print(f"  α (alpha) = {alpha}")
    if regularization_type == 'elasticnet':
        print(f"  ρ (l1_ratio) = {l1_ratio}")


def get_regularization_config():
    """Return current regularization configuration."""
    return REGULARIZATION_CONFIG.copy()


def make_feature_vector(data, include_velocity_ratio=False, include_trig_features=False, include_kinetic_features=False):
    """
    Constructs a 2D numpy array from a list of dictionaries containing feature values.

    Parameters:
        data (list of dict): Each dict represents a state with keys 'x', 'y', 'v_x', 'v_y'.
        include_velocity_ratio (bool): If True, adds velocity_ratio as feature.
            velocity_ratio = |v_y| / (|v_x| + |v_y|) - captures impact angle.
        include_trig_features (bool): If True, adds cos, sin, tan of velocity angle.
            These capture the impact angle in trigonometric form.
        include_kinetic_features (bool): If True, adds speed (v), v_x², v_y².
            These capture kinetic energy-related properties.

    Returns:
        np.ndarray: Array with features based on flags:
            Base features: [x, y, v_x, v_y] (4 features)
            + velocity_ratio: adds 1 feature (5 total)
            + trig_features: adds 3 features (cos, sin, tan)
            + kinetic_features: adds 3 features (speed, v_x², v_y²)
            All features: [x, y, v_x, v_y, velocity_ratio, cos, sin, tan, speed, v_x², v_y²] (11 features)
    """
    result = []
    for sample in data:
        row = [sample['x'], sample['y'], sample['v_x'], sample['v_y']]
        if include_velocity_ratio:
            ratio = compute_velocity_ratio(sample['v_x'], sample['v_y'])
            row.append(ratio)
        if include_trig_features:
            cos_ang, sin_ang, tan_ang = compute_angle_trig_features(sample['v_x'], sample['v_y'])
            row.extend([cos_ang, sin_ang, tan_ang])
        if include_kinetic_features:
            speed, vx_sq, vy_sq = compute_kinetic_features(sample['v_x'], sample['v_y'])
            row.extend([speed, vx_sq, vy_sq])
        result.append(row)
    return np.array(result)


ROLLING_LR_MIN_SAMPLES = 3
ROLLING_LR_OUTPUT_VARS = ("x", "y", "v_x", "v_y")
# Extended inputs: impact angle (ratio + trig) and kinetic (speed, v_x², v_y²).
ROLLING_LR_FEATURE_FLAGS = {
    "include_velocity_ratio": True,
    "include_trig_features": True,
    "include_kinetic_features": True,
}
# Base 4-feature LR — injectable into PDDL (ENHSP cannot express extended features).
ROLLING_LR_PDDL_FEATURE_FLAGS = {
    "include_velocity_ratio": False,
    "include_trig_features": False,
    "include_kinetic_features": False,
}


def rolling_lr_feature_flags(lr_models: dict = None) -> dict:
    """Feature flags for rolling LR X matrix; stored models override defaults."""
    if isinstance(lr_models, dict):
        stored = lr_models.get("_feature_flags")
        if isinstance(stored, dict):
            return stored
    return ROLLING_LR_FEATURE_FLAGS.copy()


def rolling_lr_uses_extended_features(lr_models: dict = None) -> bool:
    flags = rolling_lr_feature_flags(lr_models)
    return any(
        flags.get(key)
        for key in (
            "include_velocity_ratio",
            "include_trig_features",
            "include_kinetic_features",
        )
    )


def make_rolling_lr_feature_vector(states, lr_models: dict = None):
    flags = rolling_lr_feature_flags(lr_models)
    return make_feature_vector(states, **flags)


def train_rolling_lr(
    kb_event: dict,
    min_samples: int = ROLLING_LR_MIN_SAMPLES,
    feature_flags: dict = None,
):
    """
    Fit linear regression for surface rolling.

    X = [x, y, v_x, v_y] plus optional impact-angle and kinetic derived features.
    Y = post-contact x, y, v_x, v_y (one model per output).
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    states = kb_event.get("states") or []
    variables = kb_event.get("variables") or {}
    if len(states) < min_samples:
        return None

    if feature_flags is None:
        feature_flags = rolling_lr_feature_flags()
    else:
        feature_flags = feature_flags.copy()
    X = make_feature_vector(states, **feature_flags)
    lr_models = {"_feature_flags": feature_flags.copy()}
    for var_name in ROLLING_LR_OUTPUT_VARS:
        if var_name not in variables:
            return None
        y = np.array(variables[var_name]["value"])
        if len(y) != len(X):
            return None
        model = LinearRegression()
        model.fit(X, y)
        r2 = float(r2_score(y, model.predict(X)))
        lr_models[var_name] = {
            "model": model,
            "r2": r2,
            "coef": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "n_features": int(X.shape[1]),
        }

    if len([k for k in lr_models if k in ROLLING_LR_OUTPUT_VARS]) != len(ROLLING_LR_OUTPUT_VARS):
        return None
    return lr_models


def train_rolling_lr_dual(kb_event: dict, min_samples: int = ROLLING_LR_MIN_SAMPLES):
    """Train extended LR for forward sim and base 4-feature LR for PDDL injection."""
    sim_models = train_rolling_lr(
        kb_event, min_samples, feature_flags=ROLLING_LR_FEATURE_FLAGS.copy(),
    )
    pddl_models = train_rolling_lr(
        kb_event, min_samples, feature_flags=ROLLING_LR_PDDL_FEATURE_FLAGS.copy(),
    )
    return sim_models, pddl_models


# Platform slide v_x for PDDL: linear 4-feature LR collapses (~0.04 R²) when KB mixes
# bounces and slides. Ratio on slide-like samples is injectable and stable.
PLATFORM_SLIDE_POST_VY_MAX = 15.0
PLATFORM_VX_RATIO_MIN_SAMPLES = 5
PLATFORM_VX_RATIO_MIN_PRE_VX = 25.0
# Keep samples that look like forward slide (not reverse bounce / sign flip).
PLATFORM_VX_RATIO_MIN_RETENTION = 0.35
PLATFORM_VX_RATIO_MAX_RETENTION = 1.05


def _platform_vx_ratio_sample_indices(kb_event: dict, slide_only: bool = True):
    states = kb_event.get("states") or []
    variables = kb_event.get("variables") or {}
    post_vy = variables.get("v_y", {}).get("value") or []
    post_vx = variables.get("v_x", {}).get("value") or []
    indices = []
    for i, st in enumerate(states):
        if i >= len(post_vy) or i >= len(post_vx):
            break
        pre_vx = float(st.get("v_x", 0.0))
        if abs(pre_vx) < PLATFORM_VX_RATIO_MIN_PRE_VX:
            continue
        pv = float(post_vx[i])
        vy = float(post_vy[i])
        if slide_only:
            if abs(vy) > PLATFORM_SLIDE_POST_VY_MAX:
                continue
            if pv != 0.0 and np.sign(pv) != np.sign(pre_vx):
                continue
            ratio = pv / pre_vx if pre_vx != 0.0 else 0.0
            if not (PLATFORM_VX_RATIO_MIN_RETENTION <= abs(ratio) <= PLATFORM_VX_RATIO_MAX_RETENTION):
                continue
        indices.append(i)
    return indices


def train_platform_pddl_vx_ratio(kb_event: dict, slide_only: bool = True):
    """
    Fit injectable v_x = ratio * pre_vx for platform PDDL slide entry.

    Uses slide-like samples (|post_v_y| small) when available; otherwise all samples
    with meaningful pre_v_x. Returns an lr_models['v_x'] entry or None.
    """
    indices = _platform_vx_ratio_sample_indices(kb_event, slide_only=slide_only)
    if len(indices) < PLATFORM_VX_RATIO_MIN_SAMPLES:
        return None

    states = kb_event["states"]
    variables = kb_event["variables"]
    pre_vx = [float(states[i]["v_x"]) for i in indices]
    post_vx = [float(variables["v_x"]["value"][i]) for i in indices]
    filtered_states = [states[i] for i in indices]

    ratio_model = PhysicsRatioModel("v_x")
    ratio_model.fit(pre_vx, post_vx)
    if ratio_model.ratio is None:
        return None

    from sklearn.metrics import r2_score

    X = make_feature_vector(filtered_states, **ROLLING_LR_PDDL_FEATURE_FLAGS)
    y = np.array(post_vx, dtype=float)
    pred = ratio_model.predict(X)
    r2 = float(r2_score(y, pred)) if len(y) >= 2 else 0.0

    return {
        "model": ratio_model,
        "r2": r2,
        "coef": ratio_model.coef_.tolist(),
        "intercept": float(ratio_model.intercept_),
        "n_features": 4,
        "_vx_model_kind": "ratio_slide" if slide_only else "ratio_all",
        "_vx_ratio_n": len(indices),
    }


def refresh_platform_pddl_vx_ratio(kb_event: dict) -> bool:
    """Patch lr_models_pddl v_x with slide-filtered ratio model when possible."""
    if not isinstance(kb_event, dict):
        return False
    pddl_lr = kb_event.get("lr_models_pddl")
    if not isinstance(pddl_lr, dict):
        return False
    if not all(var in pddl_lr for var in ROLLING_LR_OUTPUT_VARS):
        return False

    vx_entry = train_platform_pddl_vx_ratio(kb_event, slide_only=True)
    if vx_entry is None:
        vx_entry = train_platform_pddl_vx_ratio(kb_event, slide_only=False)
    if vx_entry is None:
        return False

    pddl_lr = dict(pddl_lr)
    pddl_lr["v_x"] = vx_entry
    kb_event["lr_models_pddl"] = pddl_lr
    return True


def update_model_effects(
    event_name: str,
    kb: dict,
    pre_event_state: dict,
    post_event_state: dict,
    train_level: int = None,
    debug: bool = False,
):
    """
    Updates the knowledge base with a new event and retrains M5 collision models.

    Samples are appended every call; full M5 retrain runs on the first sample and
    then every COLLISION_RETRAIN_EVERY_N_LEVELS completed train levels (at most
    once per milestone level).

    Parameters:
        event_name (str): Name of the event.
        kb (dict): Knowledge base dictionary.
        pre_event_state (dict): Dictionary of features before the event.
        post_event_state (dict): Dictionary of features after the event.
        train_level (int): Current train level index (1-based phy-q level counter).
        debug (bool): If True, print model comparison debug output.
    
    Returns:
        EventModelManager: The manager instance with comparison results (for debug access).
    """
    # Initialize event in KB if not present
    if event_name not in kb:
        kb[event_name] = {
            "states": [pre_event_state],
            "variables": {
                var: {"value": [post_event_state[var]], "model": None, "model_comparison": None}
                for var in post_event_state.keys()
            }
        }
    else:
        kb[event_name]["states"].append(pre_event_state)
        for var in ROLLING_LR_OUTPUT_VARS:
            if var not in kb[event_name]["variables"]:
                kb[event_name]["variables"][var] = {
                    "value": [], "model": None, "model_comparison": None
                }
        for var in post_event_state:
            if var in kb[event_name]["variables"]:
                kb[event_name]["variables"][var]["value"].append(post_event_state[var])

    n_samples = len(kb[event_name]["states"])

    sim_lr, pddl_lr = train_rolling_lr_dual(kb[event_name])
    if sim_lr:
        kb[event_name]["lr_models"] = sim_lr
        r2_parts = ", ".join(
            f"{var}={sim_lr[var]['r2']:.3f}" for var in ROLLING_LR_OUTPUT_VARS
        )
        n_feat = sim_lr.get("x", {}).get("n_features", 4)
        feat_note = "extended" if rolling_lr_uses_extended_features(sim_lr) else "base"
        print(
            f"[ROLL LR] {event_name}: n={n_samples}  feats={n_feat} ({feat_note})  "
            f"R²=({r2_parts})"
        )
    if pddl_lr:
        kb[event_name]["lr_models_pddl"] = pddl_lr
        if event_name == "platform_collision":
            refresh_platform_pddl_vx_ratio(kb[event_name])
            pddl_lr = kb[event_name]["lr_models_pddl"]
        pddl_r2 = ", ".join(
            f"{var}={pddl_lr[var]['r2']:.3f}" for var in ROLLING_LR_OUTPUT_VARS
        )
        n_pddl_feat = pddl_lr.get("x", {}).get("n_features", 4)
        vx_note = ""
        vx_entry = pddl_lr.get("v_x") or {}
        if vx_entry.get("_vx_model_kind"):
            vx_note = (
                f"  v_x={vx_entry['_vx_model_kind']}"
                f"(n={vx_entry.get('_vx_ratio_n', '?')})"
            )
        print(
            f"[ROLL LR PDDL] {event_name}: n={n_samples}  feats={n_pddl_feat} (base)  "
            f"R²=({pddl_r2}){vx_note}"
        )

    should_retrain = n_samples == 1
    if not should_retrain and train_level is not None:
        last_level = kb.get("_m5_last_retrain_level", -1)
        if (
            train_level > 0
            and train_level % COLLISION_RETRAIN_EVERY_N_LEVELS == 0
            and train_level != last_level
        ):
            should_retrain = True
    if not should_retrain:
        defer_note = f"every {COLLISION_RETRAIN_EVERY_N_LEVELS} train levels"
        if train_level is not None:
            defer_note += f" (level {train_level})"
        print(
            f"[COLLISION-LEARN] {event_name}: stored sample {n_samples}, "
            f"retrain deferred ({defer_note})"
        )
        return EventModelManager()

    if train_level is not None:
        kb["_m5_last_retrain_level"] = train_level

    # Create feature matrix and get pre-states
    states = make_feature_vector(kb[event_name]["states"])
    pre_states = kb[event_name]["states"]
    
    print(
        f"[COLLISION-LEARN] {event_name}: retraining M5 on {n_samples} sample(s)"
    )

    # Use EventModelManager for M5 training
    manager = EventModelManager()
    
    for var_name, variable in kb[event_name]["variables"].items():
        y = np.array(variable["value"])
        
        # Train and compare models
        result = manager.train_and_compare(
            event_name=event_name,
            var_name=var_name,
            X=states,
            y=y,
            pre_states=pre_states
        )
        
        variable["model"] = result['m5_model']
        variable["model_comparison"] = result
        
        m5_hist_keys = [f"m5_{r}" for r in M5_LEAF_REG_ORDER]
        
        # Track LOO-CV history for plotting
        if "loo_cv_history" not in variable:
            variable["loo_cv_history"] = {
                "general": [],
                "general_std": [],
                "cart": [],
                **{k: [] for k in m5_hist_keys},
                "n_samples": []
            }
        
        hist = variable["loo_cv_history"]
        n_existing = len(hist["n_samples"])
        
        # Migrate legacy single "m5" series → m5_l1; add per-reg keys with inf padding
        if "m5" in hist and "m5_l1" not in hist:
            hist["m5_l1"] = list(hist["m5"])
            del hist["m5"]
        for k in m5_hist_keys:
            if k not in hist:
                hist[k] = [float('inf')] * n_existing
        
        # Append current LOO-CV values
        general_loo = result['general_stats']['loo_cv'] if result['general_stats'] else float('inf')
        general_std = result['general_stats'].get('loo_cv_std', 0) if result['general_stats'] else 0
        cart_loo = result['cart_stats']['loo_cv'] if result['cart_stats'] else float('inf')
        n_samples = len(y)
        
        m5_loo_parts = []
        if result.get('m5_stats_by_reg'):
            for reg in M5_LEAF_REG_ORDER:
                st = result['m5_stats_by_reg'][reg]
                v = st['loo_cv'] if st else float('inf')
                m5_loo_parts.append(f"{reg}={v:.4f}")
                hist[f"m5_{reg}"].append(v)
        else:
            m5_loo = result['m5_stats']['loo_cv'] if result.get('m5_stats') else float('inf')
            m5_loo_parts.append(f"legacy={m5_loo:.4f}")
            hist["m5_l1"].append(m5_loo)
            for k in m5_hist_keys:
                if k != "m5_l1":
                    hist[k].append(float('inf'))
        
        # COMMENTED OUT: LOO-CV debug output for M5/CART comparison
        # print(f"[LOO-CV DEBUG] {var_name}: General={general_loo:.4f}, CART={cart_loo:.4f}, M5[{', '.join(m5_loo_parts)}]")
        
        hist["general"].append(general_loo)
        hist["general_std"].append(general_std)
        hist["cart"].append(cart_loo)
        hist["n_samples"].append(n_samples)
        
        # COMMENTED OUT: M5 vs CART comparison printing
        # cart_model = result['cart_model']
        # if cart_model is not None and hasattr(cart_model, 'print_tree_rules'):
        #     cart_model.print_tree_rules()
        # 
        # if result.get('m5_models'):
        #     for reg in M5_LEAF_REG_ORDER:
        #         m = result['m5_models'][reg]
        #         if m is not None and hasattr(m, 'print_tree_rules'):
        #             m.print_tree_rules()
        # else:
        #     m5_model = result.get('m5_model')
        #     if m5_model is not None and hasattr(m5_model, 'print_tree_rules'):
        #         m5_model.print_tree_rules()
    
    # Print debug output if requested
    if debug:
        print(manager.get_debug_output())
    
    # COMMENTED OUT: Multiple alpha/regularization comparison printing
    # config = REGULARIZATION_CONFIG
    # for var_name in ['v_x', 'v_y']:
    #     if var_name in kb[event_name]["variables"]:
    #         y = np.array(kb[event_name]["variables"][var_name]["value"])
    #         print_all_regularization_equations(
    #             states, y, var_name,
    #             alpha=config['alpha'],
    #             l1_ratio=config['l1_ratio']
    #         )
    
    return manager


def update_model_effects_with_ablation(event_name: str, kb: dict, pre_event_state: dict, post_event_state: dict, debug: bool = False):
    """
    Updates knowledge base and runs ablation study comparing models with/without velocity_ratio.
    
    This function:
    1. Calls the standard update_model_effects() for normal model training
    2. Additionally runs an A/B comparison (ablation study) to validate if
       adding velocity_ratio as a feature improves model performance
    
    Parameters:
        event_name (str): Name of the event (e.g., "collision")
        kb (dict): Knowledge base dictionary
        pre_event_state (dict): Dictionary of features before the event
        post_event_state (dict): Dictionary of features after the event
        debug (bool): If True, print detailed debug output
    
    Returns:
        dict with:
        - manager: EventModelManager from standard training
        - ablation_results: dict of ablation comparison results per variable
    """
    # First, run the standard model training
    manager = update_model_effects(event_name, kb, pre_event_state, post_event_state, debug)
    
    # Initialize ablation storage in KB if not present
    if "ablation" not in kb[event_name]:
        kb[event_name]["ablation"] = {
            "multi_feature": {
                "results_by_var": {},
                "history": {
                    "v_x": {},
                    "v_y": {},
                    "y": {}
                }
            }
        }
    
    # Create feature matrices for all combinations
    # Feature sets to compare:
    # 1. base: [x, y, v_x, v_y] - 4 features
    # 2. ratio: [x, y, v_x, v_y, velocity_ratio] - 5 features  
    # 3. trig: [x, y, v_x, v_y, cos, sin, tan] - 7 features
    # 4. kinetic: [x, y, v_x, v_y, speed, v_x², v_y²] - 7 features
    # 5. all: [x, y, v_x, v_y, ratio, cos, sin, tan, speed, v_x², v_y²] - 11 features
    states_base = make_feature_vector(kb[event_name]["states"], 
                                      include_velocity_ratio=False, include_trig_features=False, include_kinetic_features=False)
    states_ratio = make_feature_vector(kb[event_name]["states"], 
                                       include_velocity_ratio=True, include_trig_features=False, include_kinetic_features=False)
    states_trig = make_feature_vector(kb[event_name]["states"], 
                                      include_velocity_ratio=False, include_trig_features=True, include_kinetic_features=False)
    states_kinetic = make_feature_vector(kb[event_name]["states"], 
                                         include_velocity_ratio=False, include_trig_features=False, include_kinetic_features=True)
    states_all = make_feature_vector(kb[event_name]["states"], 
                                     include_velocity_ratio=True, include_trig_features=True, include_kinetic_features=True)
    
    feature_sets = {
        'base': states_base,           # [x, y, v_x, v_y]
        'ratio': states_ratio,         # [x, y, v_x, v_y, velocity_ratio]
        'trig': states_trig,           # [x, y, v_x, v_y, cos, sin, tan]
        'kinetic': states_kinetic,     # [x, y, v_x, v_y, speed, v_x², v_y²]
        'all': states_all              # [x, y, v_x, v_y, ratio, cos, sin, tan, speed, v_x², v_y²]
    }
    
    feature_names = {
        'base': '[x, y, v_x, v_y]',
        'ratio': '[x, y, v_x, v_y, ratio]',
        'trig': '[x, y, v_x, v_y, cos, sin, tan]',
        'kinetic': '[x, y, v_x, v_y, v, v_x², v_y²]',
        'all': '[x, y, v_x, v_y, ratio, trig, v, v_x², v_y²]'
    }
    
    ablation_results = {}
    ablation_manager = EventModelManager()
    
    print("\n" + "=" * 110)
    print("ABLATION STUDY: Feature Set Comparison")
    print("=" * 110)
    print(f"{'Variable':<8} {'Feature Set':<40} {'LOO-CV':<12} {'vs Base':<12} {'Best?'}")
    print("-" * 110)
    
    # Run ablation comparison for each target variable
    for var_name in ['v_x', 'v_y', 'y']:
        if var_name not in kb[event_name]["variables"]:
            continue
            
        y = np.array(kb[event_name]["variables"][var_name]["value"])
        
        # Run multi-feature comparison
        result = ablation_manager.train_multi_feature_comparison(
            var_name=var_name,
            feature_sets=feature_sets,
            y=y
        )
        
        ablation_results[var_name] = result
        kb[event_name]["ablation"]["multi_feature"]["results_by_var"][var_name] = result
        
        # Update history
        history = kb[event_name]["ablation"]["multi_feature"]["history"][var_name]
        for set_name, set_result in result.items():
            if set_name not in history:
                history[set_name] = {"loo_cv": [], "improvement_vs_base": [], "n_samples": []}
            history[set_name]["loo_cv"].append(set_result['stats']['loo_cv'])
            history[set_name]["improvement_vs_base"].append(set_result['improvement_vs_base'])
            history[set_name]["n_samples"].append(set_result['n_samples'])
        
        # Print results for this variable
        n_samples = result.get('base', {}).get('n_samples', 0)
        if n_samples < 3:
            print(f"{var_name:<8} (insufficient data: {n_samples} samples, need >= 3)")
            print("-" * 110)
            continue
            
        for set_name in ['base', 'ratio', 'trig', 'kinetic', 'all']:
            if set_name not in result:
                continue
            r = result[set_name]
            loo = r['stats']['loo_cv']
            improvement = r['improvement_vs_base']
            is_best = r['is_best']
            
            loo_str = f"{loo:.4f}" if np.isfinite(loo) else "N/A"
            imp_str = f"{improvement:+.1f}%" if np.isfinite(improvement) and set_name != 'base' else "-"
            best_str = "★ BEST" if is_best else ""
            
            print(f"{var_name:<8} {feature_names[set_name]:<40} {loo_str:<12} {imp_str:<12} {best_str}")
        print("-" * 110)
    
    # Summary: find overall best feature set
    print("\n" + "=" * 110)
    print("SUMMARY: Best Feature Set per Variable")
    print("=" * 110)
    
    best_counts = {'base': 0, 'ratio': 0, 'trig': 0, 'kinetic': 0, 'all': 0}
    has_valid_results = False
    for var_name, var_results in ablation_results.items():
        for set_name, result in var_results.items():
            # Only count if we have valid data (not inf LOO-CV)
            loo_cv = result['stats']['loo_cv']
            if result['is_best'] and np.isfinite(loo_cv):
                has_valid_results = True
                best_counts[set_name] = best_counts.get(set_name, 0) + 1
                print(f"  {var_name}: {feature_names[set_name]} (LOO-CV: {loo_cv:.4f})")
    
    if not has_valid_results:
        print("  (insufficient data - need >= 3 collision samples)")
        overall_best = 'base'  # Default to base when no valid data
    else:
        # Overall recommendation
        overall_best = max(best_counts.keys(), key=lambda k: best_counts[k])
    
    print(f"\n[ABLATION] OVERALL RECOMMENDATION: {feature_names[overall_best]}")
    print(f"[ABLATION] Wins: base={best_counts['base']}, ratio={best_counts['ratio']}, trig={best_counts['trig']}, kinetic={best_counts['kinetic']}, all={best_counts['all']}")
    print("=" * 110 + "\n")
    
    return {
        'manager': manager,
        'ablation_results': ablation_results,
        'ablation_manager': ablation_manager,
        'best_feature_set': overall_best,
        'feature_names': feature_names
    }


def analyze_event_clusters(kb, event_name="collision", n_clusters=2, plot=True, save_path=None):
    """
    Analyze if collision events form distinct clusters based on impact angle.
    
    Uses K-Means clustering to identify if there are 2 (or more) distinct
    collision behaviors, then compares:
    1. Single model for all data
    2. Separate models per cluster
    
    Parameters:
        kb: Knowledge base dictionary
        event_name: Name of event to analyze (default: "collision")
        n_clusters: Number of clusters to test (default: 2)
        plot: Whether to show visualization (default: True)
        save_path: Optional path to save the plot
    
    Returns:
        dict with cluster analysis results
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib
    if not plot:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if event_name not in kb:
        print(f"[CLUSTER ANALYSIS] No '{event_name}' event in KB")
        return None
    
    states = kb[event_name]["states"]
    n_samples = len(states)
    
    if n_samples < 4:
        print(f"[CLUSTER ANALYSIS] Need at least 4 samples for clustering (have {n_samples})")
        return None
    
    # Extract features for clustering
    # Key features: impact_angle_factor, v_y magnitude, v_x magnitude
    features = []
    impact_factors = []
    launch_angles_approx = []  # Approximate launch angle from velocity ratio
    
    for state in states:
        v_x = state['v_x']
        v_y = state['v_y']
        total_speed = abs(v_x) + abs(v_y)
        
        # Impact angle factor: 0 = horizontal, 1 = vertical
        if total_speed > 0:
            impact_factor = abs(v_y) / total_speed
        else:
            impact_factor = 0.5
        
        impact_factors.append(impact_factor)
        
        # Approximate launch angle (using atan2 of velocity components)
        import math
        if abs(v_x) > 0.1:
            angle_approx = math.degrees(math.atan(abs(v_y) / abs(v_x)))
        else:
            angle_approx = 90  # Nearly vertical
        launch_angles_approx.append(angle_approx)
        
        # Feature vector: [impact_factor, |v_y|, |v_x|]
        features.append([impact_factor, abs(v_y), abs(v_x)])
    
    X = np.array(features)
    
    # Normalize features for clustering
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different numbers of clusters and compute silhouette scores
    silhouette_scores = {}
    cluster_labels = {}
    
    for k in range(2, min(n_samples - 1, 5)):  # Test 2, 3, 4 clusters
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
            score = silhouette_score(X_scaled, labels)
            silhouette_scores[k] = score
            cluster_labels[k] = labels
    
    # Find optimal number of clusters
    if silhouette_scores:
        best_k = max(silhouette_scores, key=silhouette_scores.get)
        best_score = silhouette_scores[best_k]
    else:
        best_k = 2
        best_score = 0
    
    # Use requested n_clusters for analysis
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Analyze each cluster
    cluster_stats = {}
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        cluster_states = [states[i] for i in cluster_indices]
        cluster_impact_factors = [impact_factors[i] for i in cluster_indices]
        cluster_angles = [launch_angles_approx[i] for i in cluster_indices]
        
        # Get post-collision values for this cluster
        v_y_post = [kb[event_name]["variables"]["v_y"]["value"][i] for i in cluster_indices]
        v_x_post = [kb[event_name]["variables"]["v_x"]["value"][i] for i in cluster_indices]
        
        # Calculate ratios
        v_y_ratios = []
        v_x_ratios = []
        for i, state in enumerate(cluster_states):
            if abs(state['v_y']) > 0.1:
                v_y_ratios.append(v_y_post[i] / state['v_y'])
            if abs(state['v_x']) > 0.1:
                v_x_ratios.append(v_x_post[i] / state['v_x'])
        
        cluster_stats[cluster_id] = {
            'n_samples': len(cluster_indices),
            'indices': cluster_indices.tolist(),
            'impact_factor_mean': np.mean(cluster_impact_factors),
            'impact_factor_std': np.std(cluster_impact_factors),
            'impact_factor_range': [min(cluster_impact_factors), max(cluster_impact_factors)],
            'angle_mean': np.mean(cluster_angles),
            'angle_range': [min(cluster_angles), max(cluster_angles)],
            'v_y_ratio_mean': np.mean(v_y_ratios) if v_y_ratios else 0,
            'v_y_ratio_std': np.std(v_y_ratios) if len(v_y_ratios) > 1 else 0,
            'v_x_ratio_mean': np.mean(v_x_ratios) if v_x_ratios else 0,
            'v_x_ratio_std': np.std(v_x_ratios) if len(v_x_ratios) > 1 else 0,
        }
    
    # Compare single model vs cluster-specific models (using LOO-CV)
    single_model_loo = _compute_single_model_loo(kb, event_name)
    cluster_model_loo = _compute_cluster_model_loo(kb, event_name, labels, n_clusters)
    
    improvement_v_y = ((single_model_loo['v_y'] - cluster_model_loo['v_y']) / single_model_loo['v_y'] * 100) if single_model_loo['v_y'] > 0 else 0
    improvement_v_x = ((single_model_loo['v_x'] - cluster_model_loo['v_x']) / single_model_loo['v_x'] * 100) if single_model_loo['v_x'] > 0 else 0
    
    # Print results
    print("\n" + "=" * 80)
    print("CLUSTER ANALYSIS FOR COLLISION EVENTS")
    print("=" * 80)
    
    print(f"\n📊 SILHOUETTE SCORES (higher = better separation):")
    for k, score in silhouette_scores.items():
        marker = " ← BEST" if k == best_k else ""
        print(f"   {k} clusters: {score:.3f}{marker}")
    
    if best_score < 0.3:
        print(f"\n⚠️  Low silhouette score ({best_score:.3f}) suggests data may NOT have clear clusters")
    elif best_score < 0.5:
        print(f"\n🔶 Moderate silhouette score ({best_score:.3f}) suggests some cluster structure")
    else:
        print(f"\n✅ Good silhouette score ({best_score:.3f}) suggests clear cluster structure")
    
    print(f"\n📈 CLUSTER STATISTICS (using {n_clusters} clusters):")
    print("-" * 80)
    
    for cluster_id, stats in cluster_stats.items():
        angle_type = "STEEP" if stats['impact_factor_mean'] > 0.5 else "SHALLOW"
        print(f"\n  Cluster {cluster_id} ({angle_type} impacts): {stats['n_samples']} samples")
        print(f"    Impact factor: {stats['impact_factor_mean']:.3f} ± {stats['impact_factor_std']:.3f}")
        print(f"    Approx angle range: {stats['angle_range'][0]:.1f}° - {stats['angle_range'][1]:.1f}°")
        print(f"    v_y ratio (restitution): {stats['v_y_ratio_mean']:.3f} ± {stats['v_y_ratio_std']:.3f}")
        print(f"    v_x ratio (friction): {stats['v_x_ratio_mean']:.3f} ± {stats['v_x_ratio_std']:.3f}")
    
    print(f"\n📉 MODEL COMPARISON (LOO-CV RMSE):")
    print("-" * 80)
    print(f"  {'Model':<30} {'v_y RMSE':<15} {'v_x RMSE':<15}")
    print(f"  {'-'*60}")
    print(f"  {'Single model (all data)':<30} {single_model_loo['v_y']:<15.4f} {single_model_loo['v_x']:<15.4f}")
    print(f"  {'Cluster-specific models':<30} {cluster_model_loo['v_y']:<15.4f} {cluster_model_loo['v_x']:<15.4f}")
    print(f"  {'-'*60}")
    print(f"  {'Improvement':<30} {improvement_v_y:>+14.1f}% {improvement_v_x:>+14.1f}%")
    
    if improvement_v_y > 10 or improvement_v_x > 10:
        print(f"\n✅ SIGNIFICANT IMPROVEMENT: Consider using cluster-specific models!")
    elif improvement_v_y > 0 or improvement_v_x > 0:
        print(f"\n🔶 MINOR IMPROVEMENT: Clusters may help but gains are small")
    else:
        print(f"\n❌ NO IMPROVEMENT: Single model is sufficient")
    
    print("=" * 80)
    
    # Visualization
    if plot or save_path:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Collision Event Cluster Analysis', fontsize=14, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Plot 1: Impact factor vs v_y ratio
        ax1 = axes[0, 0]
        v_y_ratios_all = []
        for i, state in enumerate(states):
            if abs(state['v_y']) > 0.1:
                ratio = kb[event_name]["variables"]["v_y"]["value"][i] / state['v_y']
                v_y_ratios_all.append(ratio)
            else:
                v_y_ratios_all.append(0)
        
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            ax1.scatter(
                np.array(impact_factors)[mask],
                np.array(v_y_ratios_all)[mask],
                c=colors[cluster_id % len(colors)],
                label=f'Cluster {cluster_id}',
                alpha=0.7,
                s=80
            )
        ax1.set_xlabel('Impact Factor (0=horizontal, 1=vertical)')
        ax1.set_ylabel('v_y Ratio (restitution)')
        ax1.set_title('v_y Restitution by Impact Angle')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=-0.33, color='red', linestyle='--', alpha=0.5, label='Expected ~-0.33')
        
        # Plot 2: Impact factor vs v_x ratio
        ax2 = axes[0, 1]
        v_x_ratios_all = []
        for i, state in enumerate(states):
            if abs(state['v_x']) > 0.1:
                ratio = kb[event_name]["variables"]["v_x"]["value"][i] / state['v_x']
                v_x_ratios_all.append(ratio)
            else:
                v_x_ratios_all.append(0)
        
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            ax2.scatter(
                np.array(impact_factors)[mask],
                np.array(v_x_ratios_all)[mask],
                c=colors[cluster_id % len(colors)],
                label=f'Cluster {cluster_id}',
                alpha=0.7,
                s=80
            )
        ax2.set_xlabel('Impact Factor (0=horizontal, 1=vertical)')
        ax2.set_ylabel('v_x Ratio (friction)')
        ax2.set_title('v_x Friction by Impact Angle')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Silhouette scores
        ax3 = axes[1, 0]
        if silhouette_scores:
            ks = list(silhouette_scores.keys())
            scores = list(silhouette_scores.values())
            bars = ax3.bar(ks, scores, color=['green' if k == best_k else 'steelblue' for k in ks])
            ax3.set_xlabel('Number of Clusters')
            ax3.set_ylabel('Silhouette Score')
            ax3.set_title('Optimal Cluster Count')
            ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Good threshold')
            ax3.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Not enough data\nfor silhouette analysis',
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Model comparison
        ax4 = axes[1, 1]
        x_pos = np.arange(2)
        width = 0.35
        
        single_vals = [single_model_loo['v_y'], single_model_loo['v_x']]
        cluster_vals = [cluster_model_loo['v_y'], cluster_model_loo['v_x']]
        
        bars1 = ax4.bar(x_pos - width/2, single_vals, width, label='Single Model', color='steelblue')
        bars2 = ax4.bar(x_pos + width/2, cluster_vals, width, label='Cluster Models', color='coral')
        
        ax4.set_xlabel('Variable')
        ax4.set_ylabel('LOO-CV RMSE (lower = better)')
        ax4.set_title('Single vs Cluster Model Performance')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(['v_y', 'v_x'])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n[CLUSTER ANALYSIS] Saved plot to {save_path}")
        
        if plot:
            plt.show()
        else:
            plt.close(fig)
    
    return {
        'n_samples': n_samples,
        'n_clusters': n_clusters,
        'silhouette_scores': silhouette_scores,
        'best_k': best_k,
        'best_silhouette': best_score,
        'cluster_labels': labels.tolist(),
        'cluster_stats': cluster_stats,
        'single_model_loo': single_model_loo,
        'cluster_model_loo': cluster_model_loo,
        'improvement_v_y': improvement_v_y,
        'improvement_v_x': improvement_v_x
    }


def _compute_single_model_loo(kb, event_name):
    """Compute LOO-CV RMSE for single model on all data."""
    from sklearn.linear_model import Ridge
    
    states = make_feature_vector(kb[event_name]["states"])
    n_samples = len(states)
    
    results = {'v_y': float('inf'), 'v_x': float('inf')}
    
    for var_name in ['v_y', 'v_x']:
        y = np.array(kb[event_name]["variables"][var_name]["value"])
        
        if n_samples < 2:
            continue
        
        loo_errors = []
        for i in range(n_samples):
            X_train = np.delete(states, i, axis=0)
            y_train = np.delete(y, i)
            X_test = states[i:i+1]
            y_test = y[i]
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]
            loo_errors.append((pred - y_test) ** 2)
        
        results[var_name] = np.sqrt(np.mean(loo_errors))
    
    return results


def _compute_cluster_model_loo(kb, event_name, labels, n_clusters):
    """Compute LOO-CV RMSE using cluster-specific models."""
    from sklearn.linear_model import Ridge
    
    states = make_feature_vector(kb[event_name]["states"])
    n_samples = len(states)
    
    results = {'v_y': float('inf'), 'v_x': float('inf')}
    
    for var_name in ['v_y', 'v_x']:
        y = np.array(kb[event_name]["variables"][var_name]["value"])
        
        if n_samples < 2:
            continue
        
        loo_errors = []
        for i in range(n_samples):
            # Find which cluster this sample belongs to
            sample_cluster = labels[i]
            
            # Get indices of other samples in the same cluster
            cluster_indices = np.where(labels == sample_cluster)[0]
            train_indices = cluster_indices[cluster_indices != i]
            
            if len(train_indices) < 1:
                # Not enough samples in cluster - use all data
                X_train = np.delete(states, i, axis=0)
                y_train = np.delete(y, i)
            else:
                X_train = states[train_indices]
                y_train = y[train_indices]
            
            X_test = states[i:i+1]
            y_test = y[i]
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]
            loo_errors.append((pred - y_test) ** 2)
        
        results[var_name] = np.sqrt(np.mean(loo_errors))
    
    return results
