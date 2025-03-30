import logging
import time
from sklearn.pipeline import Pipeline, FeatureUnion
from src.utils.utils import get_memory_usage

class LoggingPipeline(Pipeline):
    """Pipeline extension that logs progress of steps."""
    def __init__(self, steps, verbose=False, logger=None):
        super().__init__(steps, verbose=verbose)
        self.logger = logger
    
    def fit(self, X, y=None, **fit_params):
        self.logger.info(f"Starting pipeline fit on {len(X)} samples")
        start_time = time.time()
        mem_before = get_memory_usage()
        
        for step_idx, (name, transform) in enumerate(self.steps[:-1]):
            self.logger.info(f"Fitting step {step_idx+1}/{len(self.steps)}: {name}")
            step_start = time.time()
            
            # Special handling for FeatureUnion to log each part
            if name == 'features' and isinstance(transform, FeatureUnion):
                for feat_name, feat_transform in transform.transformer_list:
                    feat_start = time.time()
                    self.logger.info(f"  Fitting feature extractor: {feat_name}")
                    # For GloveVectorizer, log more details
                    if 'glove' in str(feat_transform).lower():
                        self.logger.info(f"  Extracting GloVe vectors with positional encoding")
                    
            if hasattr(transform, "fit_transform"):
                if y is not None:
                    X = transform.fit_transform(X, y)
                else:
                    X = transform.fit_transform(X)
            else:
                transform.fit(X, y)
                X = transform.transform(X)
                
            step_end = time.time()
            self.logger.info(f"  Step {name} completed in {step_end - step_start:.2f} seconds")
            self.logger.info(f"  Memory after step: {get_memory_usage():.2f} MB (+ {get_memory_usage() - mem_before:.2f} MB)")
            self.logger.info(f"  Output shape: {X.shape if hasattr(X, 'shape') else 'unknown'}")
            
        # Fit the final estimator
        self.logger.info(f"Fitting final estimator: {self.steps[-1][0]}")
        final_start = time.time()
        self.steps[-1][1].fit(X, y)
        final_end = time.time()
        self.logger.info(f"Final estimator fitted in {final_end - final_start:.2f} seconds")
        
        total_time = time.time() - start_time
        self.logger.info(f"Total pipeline fit completed in {total_time:.2f} seconds")
        self.logger.info(f"Final memory usage: {get_memory_usage():.2f} MB (+ {get_memory_usage() - mem_before:.2f} MB)")
        
        return self
    
    def predict(self, X):
        self.logger.info(f"Starting prediction on {len(X)} samples")
        start_time = time.time()
        
        for step_idx, (name, transform) in enumerate(self.steps[:-1]):
            self.logger.info(f"Transforming step {step_idx+1}/{len(self.steps)-1}: {name}")
            step_start = time.time()
            X = transform.transform(X)
            step_end = time.time()
            self.logger.info(f"  Step {name} transform completed in {step_end - step_start:.2f} seconds")
        
        self.logger.info(f"Predicting with final estimator: {self.steps[-1][0]}")
        pred_start = time.time()
        y_pred = self.steps[-1][1].predict(X)
        pred_end = time.time()
        self.logger.info(f"Prediction completed in {pred_end - pred_start:.2f} seconds")
        self.logger.info(f"Total prediction time: {time.time() - start_time:.2f} seconds")
        
        return y_pred