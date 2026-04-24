"""
Organizational Network Analysis (ONA) Platform

A comprehensive toolkit for analyzing organizational networks using
machine learning and graph analytics.

The Three E's (SQLAlchemy stack, calculators, optional viz) live in the
``orgnet.three_es`` subpackage; install with the ``three-es`` extra when needed.
"""

__version__ = "1.0.0"
__author__ = "ONA Platform"

from orgnet.core import OrganizationalNetworkAnalyzer

__all__ = ["OrganizationalNetworkAnalyzer"]
