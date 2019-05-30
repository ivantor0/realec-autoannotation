import sys
import os

sys.path.insert(0, "/var/www/annotate")


from server import app as application

if __name__=="__main__":
    application.run(debug=True)
