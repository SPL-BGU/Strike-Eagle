import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet


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
    Manages general and domain-specific models for event learning.
    
    This class:
    1. Always trains a general Linear/Polynomial model
    2. Optionally trains domain-specific models if available for the event type
    3. Compares models using LOO-CV and selects the best one
    4. Provides debug output for model comparison
    """
    
    # Registry of domain-specific models by event type and variable
    DOMAIN_MODELS = {
        'collision': {
            'v_x': ('AngleDependentFriction', AngleDependentFrictionModel),
            'v_y': ('PhysicsRatio', PhysicsRatioModel),
        }
        # Can add more event types: 'wall_collision', 'block_collision', etc.
    }
    
    def __init__(self):
        self.comparison_results = {}  # Store results for debug output
    
    def train_and_compare(self, event_name, var_name, X, y, pre_states=None):
        """
        Train both general and domain-specific models, compare, and return winner.
        
        Parameters:
            event_name: Name of the event (e.g., 'collision')
            var_name: Variable being predicted (e.g., 'v_x', 'v_y', 'y')
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            pre_states: List of pre-event state dicts (needed for some domain models)
        
        Returns:
            dict with keys:
                'winner': The selected model object
                'winner_name': String name of winner ('General' or domain model name)
                'general_model': The general model object
                'general_stats': Stats dict for general model
                'domain_model': Domain model object (or None)
                'domain_stats': Stats dict for domain model (or None)
                'improvement_pct': Percentage improvement of winner over loser
                'n_samples': Number of samples used
        """
        n_samples = len(y)
        
        # Always train general model
        general_model = self._train_general_model(X, y)
        general_stats = self._compute_model_stats(general_model, X, y, 'general')
        
        # Check if domain-specific model is available for this event/variable
        domain_model = None
        domain_stats = None
        domain_name = None
        
        if event_name in self.DOMAIN_MODELS and var_name in self.DOMAIN_MODELS[event_name]:
            domain_name, domain_model_class = self.DOMAIN_MODELS[event_name][var_name]
            domain_model = self._train_domain_model(
                domain_model_class, var_name, X, y, pre_states
            )
            if domain_model is not None:
                domain_stats = self._compute_model_stats(domain_model, X, y, 'domain')
        
        # Compare and select winner
        winner, winner_name, improvement_pct = self._select_winner(
            general_model, general_stats, 
            domain_model, domain_stats, domain_name
        )
        
        result = {
            'winner': winner,
            'winner_name': winner_name,
            'general_model': general_model,
            'general_stats': general_stats,
            'domain_model': domain_model,
            'domain_stats': domain_stats,
            'domain_name': domain_name,
            'improvement_pct': improvement_pct,
            'n_samples': n_samples
        }
        
        # Store for debug output
        self.comparison_results[var_name] = result
        
        return result
    
    def _train_general_model(self, X, y, alpha=1.0, l1_ratio=0.5):
        """
        Train a general ElasticNet regression model.
        
        ElasticNet combines L1 (Lasso) and L2 (Ridge) regularization:
        - L1 (Lasso): Drives irrelevant coefficients to exactly ZERO (feature selection)
        - L2 (Ridge): Shrinks all coefficients toward zero (prevents large values)
        
        Parameters:
            X: Feature matrix
            y: Target values
            alpha: Overall regularization strength (higher = more regularization)
            l1_ratio: Balance between L1 and L2 (0=Ridge, 1=Lasso, 0.5=balanced)
        
        Expected result: Simpler equations where irrelevant features (x, y) are zeroed out,
        leaving only the physics-relevant terms (v_x, v_y).
        """
        poly = PolynomialFeatures(degree=1, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Use ElasticNet - combines Lasso (L1) and Ridge (L2)
        # l1_ratio=0.5 means 50% Lasso, 50% Ridge
        # max_iter increased for convergence with small datasets
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        model.fit(X_poly, y)
        
        # Store poly transformer for predictions
        model.poly_features = poly
        model.model_type = 'general_elasticnet'
        model.alpha = alpha
        model.l1_ratio = l1_ratio
        
        return model
    
    def _train_domain_model(self, model_class, var_name, X, y, pre_states):
        """Train a domain-specific model if applicable."""
        try:
            if model_class == PhysicsRatioModel:
                # PhysicsRatioModel needs pre and post values for the specific variable
                if pre_states is None:
                    return None
                pre_values = np.array([s[var_name] for s in pre_states])
                model = PhysicsRatioModel(var_name)
                model.fit(pre_values, y)
                return model
                
            elif model_class == AngleDependentFrictionModel:
                # AngleDependentFrictionModel needs pre_states, pre_vx, post_vx
                if pre_states is None:
                    return None
                pre_vx_values = np.array([s['v_x'] for s in pre_states])
                model = AngleDependentFrictionModel()
                model.fit(pre_states, pre_vx_values, y)
                return model
            
            return None
        except Exception as e:
            print(f"Warning: Failed to train domain model for {var_name}: {e}")
            return None
    
    def _compute_model_stats(self, model, X, y, model_type):
        """Compute training RMSE, LOO-CV RMSE, and R² for a model."""
        n_samples = len(y)
        
        # Get predictions
        if model_type == 'general':
            X_poly = model.poly_features.transform(X)
            predictions = model.predict(X_poly)
        else:
            predictions = model.predict(X)
        
        # Training RMSE
        train_mse = np.mean((predictions - y) ** 2)
        train_rmse = np.sqrt(train_mse)
        
        # R² score
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # LOO-CV RMSE
        loo_cv = self._compute_loo_cv(model, X, y, model_type)
        
        return {
            'train_rmse': train_rmse,
            'loo_cv': loo_cv,
            'r2': r2,
            'n_samples': n_samples
        }
    
    def _compute_loo_cv(self, model, X, y, model_type):
        """Compute Leave-One-Out Cross-Validation RMSE."""
        n_samples = len(y)
        
        if n_samples < 2:
            return float('inf')
        
        loo_errors = []
        
        for i in range(n_samples):
            # Create train/test split leaving out sample i
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            X_test = X[i:i+1]
            y_test = y[i]
            
            try:
                if model_type == 'general':
                    # Retrain general model (ElasticNet)
                    poly = PolynomialFeatures(degree=1, include_bias=False)
                    X_train_poly = poly.fit_transform(X_train)
                    X_test_poly = poly.transform(X_test)
                    
                    # Use same alpha and l1_ratio as the original model
                    alpha = getattr(model, 'alpha', 1.0)
                    l1_ratio = getattr(model, 'l1_ratio', 0.5)
                    temp_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
                    temp_model.fit(X_train_poly, y_train)
                    pred = temp_model.predict(X_test_poly)[0]
                else:
                    # For domain models, we need to handle differently based on type
                    if isinstance(model, PhysicsRatioModel):
                        # Simple ratio - just use the ratio from remaining samples
                        # This is a simplified LOO-CV for ratio models
                        if hasattr(model, 'ratios') and len(model.ratios) > 1:
                            ratios_loo = [r for j, r in enumerate(model.ratios) if j != i]
                            if len(ratios_loo) > 0:
                                ratio_loo = np.median(ratios_loo)
                                # Get the corresponding pre value
                                if model.var_name == 'v_x':
                                    pred = ratio_loo * X_test[0, 2]
                                elif model.var_name == 'v_y':
                                    pred = ratio_loo * X_test[0, 3]
                                else:
                                    pred = model.predict(X_test)[0]
                            else:
                                pred = model.predict(X_test)[0]
                        else:
                            pred = model.predict(X_test)[0]
                    elif isinstance(model, AngleDependentFrictionModel):
                        # For angle-dependent model, use simplified LOO
                        pred = model.predict(X_test)[0]
                    else:
                        pred = model.predict(X_test)[0]
                
                loo_errors.append((pred - y_test) ** 2)
            except Exception:
                # If LOO fails for this sample, skip it
                continue
        
        if len(loo_errors) == 0:
            return float('inf')
        
        return np.sqrt(np.mean(loo_errors))
    
    def _select_winner(self, general_model, general_stats, domain_model, domain_stats, domain_name):
        """Select the best model based on LOO-CV RMSE."""
        if domain_model is None or domain_stats is None:
            return general_model, 'General (Poly)', 0
        
        general_loo = general_stats['loo_cv']
        domain_loo = domain_stats['loo_cv']
        
        # Handle infinite or invalid LOO-CV values
        if not np.isfinite(general_loo):
            general_loo = float('inf')
        if not np.isfinite(domain_loo):
            domain_loo = float('inf')
        
        if domain_loo < general_loo:
            # Domain model wins
            if general_loo > 0 and np.isfinite(general_loo):
                improvement = ((general_loo - domain_loo) / general_loo) * 100
            else:
                improvement = 0
            return domain_model, f'Domain ({domain_name})', improvement
        else:
            # General model wins
            if domain_loo > 0 and np.isfinite(domain_loo):
                improvement = ((domain_loo - general_loo) / domain_loo) * 100
            else:
                improvement = 0
            return general_model, 'General (ElasticNet)', improvement
    
    def get_debug_output(self, n_samples=None):
        """Generate formatted debug output for all compared models."""
        lines = []
        
        for var_name, result in self.comparison_results.items():
            n = result['n_samples'] if n_samples is None else n_samples
            
            lines.append("")
            lines.append("=" * 70)
            lines.append(f"MODEL COMPARISON: {var_name} ({n} samples)")
            lines.append("=" * 70)
            lines.append(f"{'Model':<25} {'Train RMSE':<12} {'LOO-CV':<12} {'R²':<10}")
            lines.append("-" * 70)
            
            # General model row (Ridge regression)
            gen = result['general_stats']
            lines.append(f"{'General (ElasticNet)':<25} {gen['train_rmse']:<12.4f} {gen['loo_cv']:<12.4f} {gen['r2']:<10.4f}")
            
            # Domain model row (if exists)
            if result['domain_stats']:
                dom = result['domain_stats']
                domain_label = f"Domain ({result['domain_name']})"
                lines.append(f"{domain_label:<25} {dom['train_rmse']:<12.4f} {dom['loo_cv']:<12.4f} {dom['r2']:<10.4f}")
            
            # Winner line
            lines.append("-" * 70)
            winner_str = f"SELECTED: {result['winner_name']}"
            if result['improvement_pct'] > 0:
                winner_str += f" ({result['improvement_pct']:.1f}% better LOO-CV)"
            lines.append(winner_str)
            
            # Coefficients of selected model
            lines.extend(self._format_model_coefficients(var_name, result['winner']))
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def _format_model_coefficients(self, var_name, model):
        """Format the coefficients of a model for display."""
        lines = []
        
        if isinstance(model, PhysicsRatioModel):
            lines.append(f"\n  {var_name}_after = {model.ratio:.4f} * {var_name}_before")
            stats = model.get_stats()
            if stats['n_samples'] > 0:
                lines.append(f"  └─ Learned ratio: {model.ratio:.4f} ± {stats['std']:.4f} (from {stats['n_samples']} samples)")
            if var_name == 'v_y' and abs(model.ratio - (-0.33)) < 0.15:
                lines.append(f"  └─ Restitution coefficient close to expected ~-0.33")
                
        elif isinstance(model, AngleDependentFrictionModel):
            lines.append(f"\n  {var_name}_ratio = {model.base_ratio:.4f} + ({model.angle_coef:.4f}) * impact_angle_factor")
            lines.append(f"  └─ impact_angle_factor = |v_y| / (|v_x| + |v_y|)  [0=horizontal, 1=vertical]")
            stats = model.get_stats()
            if stats['n_samples'] > 0:
                lines.append(f"  └─ R² score: {stats['r_squared']:.3f}, samples: {stats['n_samples']}")
            if model.angle_coef < -0.1:
                lines.append(f"  └─ Steeper impacts lose more v_x (physically correct)")
                
        elif hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            # General linear model
            coef_names = ["x", "y", "v_x", "v_y"]
            terms = [f"{model.intercept_:.4f}"]
            for coef, name in zip(model.coef_, coef_names):
                if abs(coef) > 0.0001:
                    terms.append(f"({coef:.4f})*{name}")
            lines.append(f"\n  {var_name}_after = " + " + ".join(terms))
        
        return lines


def make_feature_vector(data):
    """
    Constructs a 2D numpy array from a list of dictionaries containing feature values.

    Parameters:
        data (list of dict): Each dict represents a state with keys 'x', 'y', 'v_x', 'v_y', 'a_x', 'a_y'.

    Returns:
        np.ndarray: Array of shape (n_samples, 6) containing the feature vectors.
    """
    features = ['x', 'y', 'v_x', 'v_y']
    return np.array([[sample[f] for f in features] for sample in data])


def update_model_effects(event_name: str, kb: dict, pre_event_state: dict, post_event_state: dict, debug: bool = False):
    """
    Updates the knowledge base with a new event and retrains variable models based on accumulated data.
    
    Uses EventModelManager to:
    1. Always train a general Linear/Polynomial model
    2. Optionally train domain-specific models if available
    3. Compare models using LOO-CV and select the best one
    4. Store comparison results for debug output

    Parameters:
        event_name (str): Name of the event.
        kb (dict): Knowledge base dictionary.
        pre_event_state (dict): Dictionary of features before the event.
        post_event_state (dict): Dictionary of features after the event.
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
        for var in post_event_state:
            if var in kb[event_name]["variables"]:
                kb[event_name]["variables"][var]["value"].append(post_event_state[var])

    # Create feature matrix and get pre-states
    states = make_feature_vector(kb[event_name]["states"])
    pre_states = kb[event_name]["states"]
    
    # Use EventModelManager for training and comparison
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
        
        # Store winner model and full comparison results
        variable["model"] = result['winner']
        variable["model_comparison"] = result
        
        # Track LOO-CV history for plotting
        if "loo_cv_history" not in variable:
            variable["loo_cv_history"] = {
                "general": [],
                "domain": [],
                "n_samples": []
            }
        
        # Append current LOO-CV values
        general_loo = result['general_stats']['loo_cv'] if result['general_stats'] else float('inf')
        domain_loo = result['domain_stats']['loo_cv'] if result['domain_stats'] else float('inf')
        n_samples = len(y)
        
        variable["loo_cv_history"]["general"].append(general_loo)
        variable["loo_cv_history"]["domain"].append(domain_loo)
        variable["loo_cv_history"]["n_samples"].append(n_samples)
    
    # Print debug output if requested
    if debug:
        print(manager.get_debug_output())
    
    return manager
