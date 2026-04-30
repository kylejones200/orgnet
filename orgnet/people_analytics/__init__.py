"""People analytics for HR insights.

This package provides predictive models and metrics for HR decision-making:
- BurnoutPredictor: Predictive burnout modeling using temporal analysis + NLP sentiment
- WorkLifeBalanceAnalyzer: Work-life balance metrics with after-hours tracking
- PreBurnoutWarningSystem: Pre-burnout warning system combining anomaly detection and burnout prediction
- TalentActivationAnalyzer: Talent activation through stretch assignments and cross-functional exposure
"""

try:
    from orgnet.people_analytics.burnout import BurnoutPredictor
    from orgnet.people_analytics.pre_burnout_warning import PreBurnoutWarningSystem
    from orgnet.people_analytics.talent_activation import TalentActivationAnalyzer
    from orgnet.people_analytics.work_life import WorkLifeBalanceAnalyzer

    __all__ = [
        "BurnoutPredictor",
        "WorkLifeBalanceAnalyzer",
        "PreBurnoutWarningSystem",
        "TalentActivationAnalyzer",
    ]
except ImportError:
    __all__ = []
