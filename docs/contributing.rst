Contributing
============

We welcome contributions! Please see our contributing guidelines.

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/kylejones200/orgnet.git
   cd orgnet
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"

Code Quality
------------

We use:

- **black** for code formatting
- **flake8** for linting
- **pytest** for testing

Run checks:

.. code-block:: bash

   make format  # Format code
   make lint    # Check style
   make test    # Run tests
   make check   # All of the above

Pull Requests
-------------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Ensure all checks pass
6. Submit a pull request

