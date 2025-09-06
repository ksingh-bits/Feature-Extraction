"""wsgi entry point
This module is the entry point for the WSGI application."""

from app import app

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8051)
