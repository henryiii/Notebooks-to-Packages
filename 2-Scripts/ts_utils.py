"""Time series utilities - drop-in replacements for skits functionality using sktime."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class AutoregressiveTransformer(BaseEstimator, TransformerMixin):
    """Create lagged features from a time series for autoregression.
    
    This transformer creates num_lags lagged versions of the input time series.
    """
    
    def __init__(self, num_lags=1):
        self.num_lags = num_lags
        self._y_train = None
    
    def fit(self, X, y=None):
        """Store the training target for creating lags."""
        if y is not None:
            self._y_train = y.copy()
        return self
    
    def transform(self, X):
        """Create lagged features from the stored training data."""
        if self._y_train is None:
            # No y available, return empty features
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
            return np.zeros((n_samples, self.num_lags))
        
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        lagged_features = []
        
        for lag in range(1, self.num_lags + 1):
            lagged = np.full(n_samples, np.nan)
            # Only copy as many values as we have in _y_train and can fit in the output
            copy_len = min(len(self._y_train) - lag, n_samples)
            if copy_len > 0:
                lagged[:copy_len] = self._y_train[lag:lag+copy_len]
            lagged_features.append(lagged.reshape(-1, 1))
        
        return np.hstack(lagged_features) if lagged_features else np.zeros((n_samples, 0))


class SeasonalTransformer(BaseEstimator, TransformerMixin):
    """Create seasonal lagged features from a time series.
    
    This transformer creates a single seasonal lag feature.
    """
    
    def __init__(self, seasonal_period=12):
        self.seasonal_period = seasonal_period
        self._y_train = None
    
    def fit(self, X, y=None):
        """Store the training target for creating seasonal lags."""
        if y is not None:
            self._y_train = y.copy()
        return self
    
    def transform(self, X):
        """Create seasonal lagged features from the stored training data."""
        if self._y_train is None:
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
            return np.zeros((n_samples, 1))
        
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        seasonal_lag = np.full(n_samples, np.nan)
        
        # Only copy as many values as we have in _y_train and can fit in the output
        if len(self._y_train) > self.seasonal_period:
            copy_len = min(len(self._y_train) - self.seasonal_period, n_samples)
            if copy_len > 0:
                seasonal_lag[:copy_len] = self._y_train[self.seasonal_period:self.seasonal_period+copy_len]
        
        return seasonal_lag.reshape(-1, 1)


class ReversibleImputer(BaseEstimator, TransformerMixin):
    """Imputer that handles missing values (wrapper around SimpleImputer)."""
    
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)
    
    def fit(self, X, y=None):
        """Fit the imputer."""
        self.imputer.fit(X)
        return self
    
    def transform(self, X):
        """Transform by imputing missing values."""
        return self.imputer.transform(X)


class ForecasterPipeline(Pipeline):
    """Pipeline for time series forecasting.
    
    This pipeline handles the special case where:
    1. The first scaler transforms y (the target)
    2. Feature extractors use the transformed y to create features
    3. Subsequent steps process the features
    4. The final step is the regressor
    """
    
    def fit(self, X, y=None):
        """Fit all steps of the pipeline."""
        if y is None:
            raise ValueError("y is required for ForecasterPipeline")
        
        # Step 1: Scale y using the first transformer (pre_scaler)
        y_scaled = self.steps[0][1].fit_transform(y.reshape(-1, 1)).ravel()
        
        # Step 2: Extract features using the feature union (which contains AR transformer)
        # The feature extractors need y_scaled to create lagged features
        features_transformer = self.steps[1][1]
        
        # Fit and transform feature extractors
        if hasattr(features_transformer, 'transformer_list'):
            # It's a FeatureUnion
            for name, trans in features_transformer.transformer_list:
                trans.fit(X, y_scaled)
            X_features = features_transformer.transform(X)
        else:
            X_features = features_transformer.fit_transform(X, y_scaled)
        
        # Step 3: Apply post-feature transformers (imputer and scaler)
        X_transformed = X_features
        for name, transformer in self.steps[2:-1]:
            X_transformed = transformer.fit_transform(X_transformed)
        
        # Step 4: Fit the final regressor
        self.steps[-1][1].fit(X_transformed, y_scaled)
        
        # Store the pre_scaler for inverse_transform during predict
        self._pre_scaler = self.steps[0][1]
        self._y_scaled_mean = np.mean(y_scaled)
        self._y_scaled_std = np.std(y_scaled)
        
        return self
    
    def predict(self, X, to_scale=False):
        """Make predictions.
        
        Args:
            X: Input features (typically timestamps)
            to_scale: If True, inverse transform predictions back to original scale
                     (ignored in this implementation as we always return scaled predictions)
        """
        # We need to use the stored y_train from the AR transformer to make predictions
        # Step 1: Extract features
        features_transformer = self.steps[1][1]
        X_features = features_transformer.transform(X)
        
        # Step 2: Apply post-feature transformers
        X_transformed = X_features
        for name, transformer in self.steps[2:-1]:
            X_transformed = transformer.transform(X_transformed)
        
        # Step 3: Predict using the regressor
        predictions = self.steps[-1][1].predict(X_transformed)
        
        return predictions
