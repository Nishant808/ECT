"""
Confidence Interval Analysis and Calibration.

This module implements empirical calibration checks to determine if real-world
mutations fall within predicted confidence intervals from Monte Carlo simulations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import binom, norm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings


@dataclass
class CalibrationConfig:
    """Configuration for confidence interval calibration analysis."""
    confidence_levels: List[float] = None
    num_bins: int = 10
    bootstrap_samples: int = 1000
    significance_level: float = 0.05
    min_samples_per_bin: int = 10
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.68, 0.90, 0.95, 0.99]


@dataclass
class CalibrationResult:
    """Result of confidence interval calibration analysis."""
    confidence_level: float
    expected_coverage: float
    empirical_coverage: float
    coverage_error: float
    calibration_score: float
    is_well_calibrated: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    bin_statistics: Dict


@dataclass
class CalibrationAnalysisResult:
    """Complete calibration analysis result."""
    calibration_results: List[CalibrationResult]
    overall_calibration_score: float
    reliability_diagram_data: Dict
    sharpness_metrics: Dict
    resolution_metrics: Dict
    brier_score: float
    analysis_metadata: Dict


class ConfidenceIntervalAnalyzer:
    """
    Analyzes confidence interval calibration for prediction uncertainty.
    
    Implements various calibration metrics including reliability diagrams,
    Brier scores, and statistical tests for proper calibration.
    """
    
    def __init__(self, config: CalibrationConfig):
        """
        Initialize confidence interval analyzer.
        
        Args:
            config: Calibration analysis configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_calibration(self,
                          predictions: np.ndarray,
                          prediction_intervals: Dict[str, np.ndarray],
                          actual_values: np.ndarray,
                          prediction_probabilities: Optional[np.ndarray] = None) -> CalibrationAnalysisResult:
        """
        Perform comprehensive calibration analysis.
        
        Args:
            predictions: Point predictions [n_samples, ...]
            prediction_intervals: Dictionary with confidence intervals
                                 Keys: confidence levels (e.g., '95%')
                                 Values: (lower_bounds, upper_bounds) tuples
            actual_values: Actual observed values [n_samples, ...]
            prediction_probabilities: Predicted probabilities for Brier score
            
        Returns:
            CalibrationAnalysisResult with comprehensive metrics
        """
        self.logger.info("Starting confidence interval calibration analysis...")
        
        # Validate inputs
        self._validate_inputs(predictions, prediction_intervals, actual_values)
        
        # Analyze calibration for each confidence level
        calibration_results = []
        
        for conf_level in self.config.confidence_levels:
            conf_key = f"{conf_level:.0%}"
            
            if conf_key in prediction_intervals:
                self.logger.info(f"Analyzing calibration for {conf_key} confidence level...")
                
                result = self._analyze_single_confidence_level(
                    predictions=predictions,
                    lower_bounds=prediction_intervals[conf_key][0],
                    upper_bounds=prediction_intervals[conf_key][1],
                    actual_values=actual_values,
                    confidence_level=conf_level
                )
                
                calibration_results.append(result)
        
        # Calculate overall metrics
        overall_score = self._calculate_overall_calibration_score(calibration_results)
        
        # Generate reliability diagram data
        reliability_data = self._generate_reliability_diagram_data(
            predictions, prediction_intervals, actual_values
        )
        
        # Calculate sharpness and resolution metrics
        sharpness_metrics = self._calculate_sharpness_metrics(prediction_intervals)
        resolution_metrics = self._calculate_resolution_metrics(
            predictions, actual_values
        )
        
        # Calculate Brier score if probabilities provided
        brier_score = None
        if prediction_probabilities is not None:
            brier_score = self._calculate_brier_score(
                prediction_probabilities, actual_values
            )
        
        # Compile results
        analysis_result = CalibrationAnalysisResult(
            calibration_results=calibration_results,
            overall_calibration_score=overall_score,
            reliability_diagram_data=reliability_data,
            sharpness_metrics=sharpness_metrics,
            resolution_metrics=resolution_metrics,
            brier_score=brier_score,
            analysis_metadata={
                'n_samples': len(predictions),
                'n_confidence_levels': len(calibration_results),
                'config': self.config
            }
        )
        
        self.logger.info("Calibration analysis completed successfully!")
        return analysis_result
    
    def _validate_inputs(self, predictions: np.ndarray, 
                        prediction_intervals: Dict, 
                        actual_values: np.ndarray):
        """Validate input arrays and intervals."""
        if predictions.shape != actual_values.shape:
            raise ValueError("Predictions and actual values must have same shape")
        
        if len(prediction_intervals) == 0:
            raise ValueError("No prediction intervals provided")
        
        # Check interval format
        for conf_level, intervals in prediction_intervals.items():
            if not isinstance(intervals, (tuple, list)) or len(intervals) != 2:
                raise ValueError(f"Intervals for {conf_level} must be (lower, upper) tuple")
            
            lower, upper = intervals
            if lower.shape != predictions.shape or upper.shape != predictions.shape:
                raise ValueError(f"Interval bounds for {conf_level} must match prediction shape")
    
    def _analyze_single_confidence_level(self,
                                       predictions: np.ndarray,
                                       lower_bounds: np.ndarray,
                                       upper_bounds: np.ndarray,
                                       actual_values: np.ndarray,
                                       confidence_level: float) -> CalibrationResult:
        """Analyze calibration for a single confidence level."""
        # Flatten arrays for analysis
        predictions_flat = predictions.flatten()
        lower_flat = lower_bounds.flatten()
        upper_flat = upper_bounds.flatten()
        actual_flat = actual_values.flatten()
        
        # Calculate empirical coverage
        in_interval = (actual_flat >= lower_flat) & (actual_flat <= upper_flat)
        empirical_coverage = np.mean(in_interval)
        
        # Expected coverage
        expected_coverage = confidence_level
        
        # Coverage error
        coverage_error = abs(empirical_coverage - expected_coverage)
        
        # Statistical test for proper calibration
        n_samples = len(actual_flat)
        n_covered = np.sum(in_interval)
        
        # Binomial test
        p_value = 2 * min(
            binom.cdf(n_covered, n_samples, expected_coverage),
            1 - binom.cdf(n_covered - 1, n_samples, expected_coverage)
        )
        
        # Confidence interval for empirical coverage
        coverage_ci = self._calculate_coverage_confidence_interval(
            n_covered, n_samples, self.config.significance_level
        )
        
        # Calibration score (0 = perfect, higher = worse)
        calibration_score = coverage_error / expected_coverage
        
        # Well-calibrated if p-value > significance level
        is_well_calibrated = p_value > self.config.significance_level
        
        # Bin statistics for reliability diagram
        bin_stats = self._calculate_bin_statistics(
            predictions_flat, lower_flat, upper_flat, actual_flat, confidence_level
        )
        
        return CalibrationResult(
            confidence_level=confidence_level,
            expected_coverage=expected_coverage,
            empirical_coverage=empirical_coverage,
            coverage_error=coverage_error,
            calibration_score=calibration_score,
            is_well_calibrated=is_well_calibrated,
            p_value=p_value,
            confidence_interval=coverage_ci,
            bin_statistics=bin_stats
        )
    
    def _calculate_coverage_confidence_interval(self,
                                              n_covered: int,
                                              n_total: int,
                                              alpha: float) -> Tuple[float, float]:
        """Calculate confidence interval for coverage probability."""
        # Wilson score interval
        z = stats.norm.ppf(1 - alpha/2)
        p = n_covered / n_total
        
        denominator = 1 + z**2 / n_total
        center = (p + z**2 / (2 * n_total)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denominator
        
        return (max(0, center - margin), min(1, center + margin))
    
    def _calculate_bin_statistics(self,
                                predictions: np.ndarray,
                                lower_bounds: np.ndarray,
                                upper_bounds: np.ndarray,
                                actual_values: np.ndarray,
                                confidence_level: float) -> Dict:
        """Calculate statistics for reliability diagram bins."""
        # Calculate interval widths as a proxy for confidence
        interval_widths = upper_bounds - lower_bounds
        
        # Create bins based on interval width (narrower = more confident)
        bin_edges = np.percentile(interval_widths, 
                                 np.linspace(0, 100, self.config.num_bins + 1))
        
        bin_stats = {
            'bin_edges': bin_edges.tolist(),
            'bin_centers': [],
            'empirical_coverage': [],
            'expected_coverage': [],
            'bin_counts': [],
            'mean_confidence': []
        }
        
        for i in range(self.config.num_bins):
            # Find samples in this bin
            if i == self.config.num_bins - 1:
                bin_mask = (interval_widths >= bin_edges[i]) & (interval_widths <= bin_edges[i + 1])
            else:
                bin_mask = (interval_widths >= bin_edges[i]) & (interval_widths < bin_edges[i + 1])
            
            if np.sum(bin_mask) >= self.config.min_samples_per_bin:
                # Calculate coverage for this bin
                bin_actual = actual_values[bin_mask]
                bin_lower = lower_bounds[bin_mask]
                bin_upper = upper_bounds[bin_mask]
                
                bin_coverage = np.mean((bin_actual >= bin_lower) & (bin_actual <= bin_upper))
                
                bin_stats['bin_centers'].append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_stats['empirical_coverage'].append(bin_coverage)
                bin_stats['expected_coverage'].append(confidence_level)
                bin_stats['bin_counts'].append(np.sum(bin_mask))
                bin_stats['mean_confidence'].append(confidence_level)  # Simplified
        
        return bin_stats
    
    def _calculate_overall_calibration_score(self, 
                                           calibration_results: List[CalibrationResult]) -> float:
        """Calculate overall calibration score across all confidence levels."""
        if not calibration_results:
            return float('inf')
        
        # Weighted average of calibration scores
        weights = [result.confidence_level for result in calibration_results]
        scores = [result.calibration_score for result in calibration_results]
        
        overall_score = np.average(scores, weights=weights)
        return overall_score
    
    def _generate_reliability_diagram_data(self,
                                         predictions: np.ndarray,
                                         prediction_intervals: Dict,
                                         actual_values: np.ndarray) -> Dict:
        """Generate data for reliability diagram visualization."""
        reliability_data = {}
        
        for conf_level in self.config.confidence_levels:
            conf_key = f"{conf_level:.0%}"
            
            if conf_key in prediction_intervals:
                lower, upper = prediction_intervals[conf_key]
                
                # Calculate interval widths
                widths = (upper - lower).flatten()
                actual_flat = actual_values.flatten()
                lower_flat = lower.flatten()
                upper_flat = upper.flatten()
                
                # Create bins based on interval width
                bin_edges = np.percentile(widths, np.linspace(0, 100, self.config.num_bins + 1))
                
                bin_data = {
                    'bin_centers': [],
                    'empirical_coverage': [],
                    'expected_coverage': conf_level,
                    'bin_counts': []
                }
                
                for i in range(self.config.num_bins):
                    if i == self.config.num_bins - 1:
                        mask = (widths >= bin_edges[i]) & (widths <= bin_edges[i + 1])
                    else:
                        mask = (widths >= bin_edges[i]) & (widths < bin_edges[i + 1])
                    
                    if np.sum(mask) >= self.config.min_samples_per_bin:
                        coverage = np.mean((actual_flat[mask] >= lower_flat[mask]) & 
                                         (actual_flat[mask] <= upper_flat[mask]))
                        
                        bin_data['bin_centers'].append((bin_edges[i] + bin_edges[i + 1]) / 2)
                        bin_data['empirical_coverage'].append(coverage)
                        bin_data['bin_counts'].append(np.sum(mask))
                
                reliability_data[conf_key] = bin_data
        
        return reliability_data
    
    def _calculate_sharpness_metrics(self, prediction_intervals: Dict) -> Dict:
        """Calculate sharpness metrics (interval width statistics)."""
        sharpness_metrics = {}
        
        for conf_level, (lower, upper) in prediction_intervals.items():
            widths = (upper - lower).flatten()
            
            sharpness_metrics[conf_level] = {
                'mean_width': np.mean(widths),
                'median_width': np.median(widths),
                'std_width': np.std(widths),
                'min_width': np.min(widths),
                'max_width': np.max(widths),
                'width_percentiles': {
                    '25%': np.percentile(widths, 25),
                    '75%': np.percentile(widths, 75),
                    '90%': np.percentile(widths, 90),
                    '95%': np.percentile(widths, 95)
                }
            }
        
        return sharpness_metrics
    
    def _calculate_resolution_metrics(self,
                                    predictions: np.ndarray,
                                    actual_values: np.ndarray) -> Dict:
        """Calculate resolution metrics (prediction accuracy)."""
        pred_flat = predictions.flatten()
        actual_flat = actual_values.flatten()
        
        # Basic accuracy metrics
        mse = np.mean((pred_flat - actual_flat) ** 2)
        mae = np.mean(np.abs(pred_flat - actual_flat))
        
        # Correlation
        correlation = np.corrcoef(pred_flat, actual_flat)[0, 1] if len(pred_flat) > 1 else 0
        
        # R-squared
        ss_res = np.sum((actual_flat - pred_flat) ** 2)
        ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        resolution_metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'correlation': correlation,
            'r_squared': r_squared,
            'bias': np.mean(pred_flat - actual_flat),
            'variance_ratio': np.var(pred_flat) / np.var(actual_flat) if np.var(actual_flat) > 0 else 0
        }
        
        return resolution_metrics
    
    def _calculate_brier_score(self,
                             predicted_probabilities: np.ndarray,
                             actual_values: np.ndarray) -> float:
        """Calculate Brier score for probabilistic predictions."""
        # Convert actual values to binary (for binary classification)
        # This is a simplified version - real implementation would depend on problem type
        
        prob_flat = predicted_probabilities.flatten()
        actual_flat = actual_values.flatten()
        
        # For regression, we can't directly calculate Brier score
        # Return a proxy metric instead
        if len(np.unique(actual_flat)) > 2:
            # Regression case - return MSE as proxy
            return np.mean((prob_flat - actual_flat) ** 2)
        else:
            # Binary case - true Brier score
            return np.mean((prob_flat - actual_flat) ** 2)
    
    def plot_reliability_diagram(self,
                               analysis_result: CalibrationAnalysisResult,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot reliability diagram for calibration analysis.
        
        Args:
            analysis_result: Calibration analysis result
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        reliability_data = analysis_result.reliability_diagram_data
        
        for i, (conf_level, data) in enumerate(reliability_data.items()):
            if i >= 4:  # Only plot first 4 confidence levels
                break
            
            ax = axes[i]
            
            # Plot reliability curve
            if data['bin_centers'] and data['empirical_coverage']:
                ax.plot(data['bin_centers'], data['empirical_coverage'], 
                       'o-', label='Empirical Coverage', markersize=6)
                ax.axhline(y=data['expected_coverage'], color='red', linestyle='--',
                          label=f'Expected Coverage ({data["expected_coverage"]:.0%})')
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            
            ax.set_xlabel('Prediction Confidence')
            ax.set_ylabel('Empirical Coverage')
            ax.set_title(f'Reliability Diagram - {conf_level}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # Hide unused subplots
        for i in range(len(reliability_data), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_calibration_report(self,
                                  analysis_result: CalibrationAnalysisResult,
                                  output_dir: str) -> str:
        """
        Generate comprehensive calibration report.
        
        Args:
            analysis_result: Calibration analysis result
            output_dir: Directory to save report
            
        Returns:
            Path to generated report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_path = output_path / "calibration_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Confidence Interval Calibration Report\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Overall Calibration Score**: {analysis_result.overall_calibration_score:.4f}\n")
            f.write(f"- **Number of Samples**: {analysis_result.analysis_metadata['n_samples']}\n")
            f.write(f"- **Confidence Levels Analyzed**: {analysis_result.analysis_metadata['n_confidence_levels']}\n\n")
            
            # Individual confidence level results
            f.write("## Confidence Level Analysis\n\n")
            
            for result in analysis_result.calibration_results:
                f.write(f"### {result.confidence_level:.0%} Confidence Level\n\n")
                f.write(f"- **Expected Coverage**: {result.expected_coverage:.3f}\n")
                f.write(f"- **Empirical Coverage**: {result.empirical_coverage:.3f}\n")
                f.write(f"- **Coverage Error**: {result.coverage_error:.3f}\n")
                f.write(f"- **Calibration Score**: {result.calibration_score:.3f}\n")
                f.write(f"- **Well Calibrated**: {'Yes' if result.is_well_calibrated else 'No'}\n")
                f.write(f"- **P-value**: {result.p_value:.4f}\n")
                f.write(f"- **Coverage 95% CI**: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]\n\n")
            
            # Sharpness metrics
            f.write("## Sharpness Analysis\n\n")
            for conf_level, metrics in analysis_result.sharpness_metrics.items():
                f.write(f"### {conf_level}\n")
                f.write(f"- **Mean Width**: {metrics['mean_width']:.4f}\n")
                f.write(f"- **Median Width**: {metrics['median_width']:.4f}\n")
                f.write(f"- **Width Std**: {metrics['std_width']:.4f}\n\n")
            
            # Resolution metrics
            f.write("## Resolution Analysis\n\n")
            res_metrics = analysis_result.resolution_metrics
            f.write(f"- **RMSE**: {res_metrics['rmse']:.4f}\n")
            f.write(f"- **MAE**: {res_metrics['mae']:.4f}\n")
            f.write(f"- **Correlation**: {res_metrics['correlation']:.4f}\n")
            f.write(f"- **R-squared**: {res_metrics['r_squared']:.4f}\n")
            f.write(f"- **Bias**: {res_metrics['bias']:.4f}\n\n")
            
            if analysis_result.brier_score is not None:
                f.write(f"- **Brier Score**: {analysis_result.brier_score:.4f}\n\n")
        
        self.logger.info(f"Calibration report saved to {report_path}")
        return str(report_path)


def analyze_prediction_calibration(predictions: np.ndarray,
                                 prediction_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                 actual_values: np.ndarray,
                                 config: Optional[CalibrationConfig] = None,
                                 output_dir: Optional[str] = None) -> CalibrationAnalysisResult:
    """
    Convenience function for calibration analysis.
    
    Args:
        predictions: Point predictions
        prediction_intervals: Confidence intervals
        actual_values: Actual observed values
        config: Calibration configuration
        output_dir: Output directory for results
        
    Returns:
        CalibrationAnalysisResult
    """
    if config is None:
        config = CalibrationConfig()
    
    analyzer = ConfidenceIntervalAnalyzer(config)
    
    result = analyzer.analyze_calibration(
        predictions=predictions,
        prediction_intervals=prediction_intervals,
        actual_values=actual_values
    )
    
    if output_dir:
        # Generate report
        analyzer.generate_calibration_report(result, output_dir)
        
        # Plot reliability diagram
        fig = analyzer.plot_reliability_diagram(result)
        fig.savefig(Path(output_dir) / "reliability_diagram.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return result