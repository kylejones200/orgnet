Data Ingestion
==============

Loading Data
------------

orgnet supports multiple data sources for organizational network analysis.

Supported Data Sources
----------------------

- **HRIS**: Employee information (departments, roles, managers)
- **Email**: Email communications
- **Slack/Teams**: Instant messaging data
- **Calendar**: Meeting and event data
- **Documents**: Document collaboration (SharePoint, Google Drive, Confluence)
- **Code**: Code repository commits and reviews

Data Format
-----------

See `DATA_FORMAT.md <https://github.com/kylejones200/orgnet/blob/main/DATA_FORMAT.md>`_ for detailed format requirements.

Example
-------

.. code-block:: python

   from orgnet.core import OrganizationalNetworkAnalyzer

   analyzer = OrganizationalNetworkAnalyzer()
   
   data_paths = {
       "hris": "data/hris.csv",
       "email": "data/email.csv",
       "slack": "data/slack.csv",
       "calendar": "data/calendar.csv",
   }
   
   analyzer.load_data(data_paths)


