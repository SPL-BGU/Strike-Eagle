import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression


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


def update_model_effects(event_name: str, kb: dict, pre_event_state: dict, post_event_state: dict):
    """
    Updates the knowledge base with a new event and retrains variable models based on accumulated data.

    Parameters:
        event_name (str): Name of the event.
        kb (dict): Knowledge base dictionary.
        pre_event_state (dict): Dictionary of features before the event.
        post_event_state (dict): Dictionary of features after the event.
    """
    if event_name not in kb:
        kb[event_name] = {
            "states": [pre_event_state],
            "variables": {
                var: {"value": [post_event_state[var]], "model": None}
                for var in post_event_state.keys()
            }
        }
    else:
        kb[event_name]["states"].append(pre_event_state)
        for var in post_event_state:
            if var in kb[event_name]["variables"]: # If var is an affected variable fof the event
                kb[event_name]["variables"][var]["value"].append(post_event_state[var])

    # Update models
    states = make_feature_vector(kb[event_name]["states"])
    for var_name, variable in kb[event_name]["variables"].items():
        y = np.array(variable["value"])
        variable["model"] = polyfit_var(states, y)
