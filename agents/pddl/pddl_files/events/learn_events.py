import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


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
        
        # Always train general model using global regularization config
        config = REGULARIZATION_CONFIG
        general_model = self._train_general_model(
            X, y, 
            alpha=config['alpha'], 
            l1_ratio=config['l1_ratio'],
            regularization=config['type']
        )
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
        poly = PolynomialFeatures(degree=1, include_bias=False)
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
                    poly = PolynomialFeatures(degree=1, include_bias=False)
                    X_train_poly = poly.fit_transform(X_train)
                    X_test_poly = poly.transform(X_test)
                    
                    # Get regularization parameters from original model
                    alpha = getattr(model, 'alpha', 1.0)
                    l1_ratio = getattr(model, 'l1_ratio', 0.5)
                    model_type_str = getattr(model, 'model_type', 'general_elasticnet')
                    
                    # Create appropriate model based on regularization type
                    if model_type_str == 'general_ols':
                        # Version 1: No regularization
                        temp_model = LinearRegression()
                    elif model_type_str == 'general_lasso':
                        # Version 2: L1 only
                        temp_model = Lasso(alpha=alpha, max_iter=10000)
                    elif model_type_str == 'general_ridge':
                        # Version 3: L2 only
                        temp_model = Ridge(alpha=alpha, max_iter=10000)
                    else:
                        # Version 4: ElasticNet (L1 + L2)
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
            return float('inf'), float('inf')
        
        rmse = np.sqrt(np.mean(loo_errors))
        # Standard error of LOO-CV (variance of individual errors)
        std_error = np.std(np.sqrt(loo_errors)) if len(loo_errors) > 1 else 0
        
        return rmse, std_error
    
    def _select_winner(self, general_model, general_stats, domain_model, domain_stats, domain_name):
        """Select the best model based on LOO-CV RMSE."""
        # Get the general model type name for display
        model_type_str = getattr(general_model, 'model_type', 'general_elasticnet')
        model_type_display = {
            'general_ols': 'General (OLS)',
            'general_lasso': 'General (Lasso/L1)',
            'general_ridge': 'General (Ridge/L2)',
            'general_elasticnet': 'General (ElasticNet/L1+L2)'
        }.get(model_type_str, 'General')
        
        if domain_model is None or domain_stats is None:
            return general_model, model_type_display, 0
        
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
            return general_model, model_type_display, improvement
    
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
            lines.append(f"MODEL COMPARISON: {var_name} ({n} samples)")
            lines.append("=" * 70)
            lines.append(f"{'Model':<25} {'Train RMSE':<12} {'LOO-CV':<12} {'R²':<10}")
            lines.append("-" * 70)
            
            # General model row
            gen = result['general_stats']
            lines.append(f"{model_type_display:<25} {gen['train_rmse']:<12.4f} {gen['loo_cv']:<12.4f} {gen['r2']:<10.4f}")
            
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
    'type': 'elasticnet',  # Options: 'none', 'l1', 'l2', 'elasticnet'
    'alpha': 1.0,          # Regularization strength
    'l1_ratio': 0.5        # For ElasticNet: 0=Ridge, 1=Lasso, 0.5=balanced
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
                "general_std": [],  # Standard error for variance bands
                "domain": [],
                "n_samples": []
            }
        
        # Append current LOO-CV values
        general_loo = result['general_stats']['loo_cv'] if result['general_stats'] else float('inf')
        general_std = result['general_stats'].get('loo_cv_std', 0) if result['general_stats'] else 0
        domain_loo = result['domain_stats']['loo_cv'] if result['domain_stats'] else float('inf')
        n_samples = len(y)
        
        variable["loo_cv_history"]["general"].append(general_loo)
        variable["loo_cv_history"]["general_std"].append(general_std)
        variable["loo_cv_history"]["domain"].append(domain_loo)
        variable["loo_cv_history"]["n_samples"].append(n_samples)
    
    # Print debug output if requested
    if debug:
        print(manager.get_debug_output())
    
    # Print all 4 regularization versions for v_x and v_y
    config = REGULARIZATION_CONFIG
    for var_name in ['v_x', 'v_y']:
        if var_name in kb[event_name]["variables"]:
            y = np.array(kb[event_name]["variables"][var_name]["value"])
            print_all_regularization_equations(
                states, y, var_name,
                alpha=config['alpha'],
                l1_ratio=config['l1_ratio']
            )
    
    return manager


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
