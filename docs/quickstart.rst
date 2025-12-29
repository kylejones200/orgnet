Quick Start
===========

This guide will help you get started with orgnet in just a few minutes.

Basic Usage
-----------

.. code-block:: python

   from orgnet.core import OrganizationalNetworkAnalyzer

   # Initialize analyzer
   analyzer = OrganizationalNetworkAnalyzer(config_path="config.yaml")

   # Load data
   data_paths = {
       "hris": "data/hris.csv",
       "email": "data/email.csv",
       "slack": "data/slack.csv",
       "calendar": "data/calendar.csv",
   }
   analyzer.load_data(data_paths)

   # Build graph
   graph = analyzer.build_graph()

   # Run analysis
   results = analyzer.analyze()

   # Generate report
   analyzer.generate_report(output_path="report.html")

Next Steps
----------

- See :doc:`user_guide/index` for detailed usage
- Check out :doc:`examples` for example notebooks
- Read :doc:`architecture` to understand the system design

