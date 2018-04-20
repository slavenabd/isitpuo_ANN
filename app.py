from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
   return  """
<!DOCTYPE html>
<head>
   <title>Deep artificial neural network and late-presenting posterior urethral valve</title>
   <link rel="stylesheet" href="http://stash.compjour.org/assets/css/foundation.css">
</head>
<body style="width: 880px; margin: 20px;">
    ANN is available at this  
    <a href="https://isitpuo-test.herokuapp.com">
        link
    </a>
</body>
"""

if __name__ == '__main__':
    app.run(use_reloader=True, debug=True)