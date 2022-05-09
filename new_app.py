from flask import Flask, render_template, request

import model_final as pyt
import pandas as pd

df = pd.read_csv ('05_Test_Data.csv')

app = Flask(__name__)

@app.route("/")
def hello():
  return render_template("index.html")


@app.route("/sub", methods=["POST"])
def submit():
  if request.method=="POST":
    cust = int(request.form.get("Customer ID"))
    late = int(request.form.get("Lateral ID"))

    df1 = df[df['Lateral_ID (i)']==late]
    df2 = df1[df1['Customer_ID (k)']==cust]
    
    l = int(df2['Lateral_ID (i)'])
    N = int(df2['N(i)'])
    x = int(df2['Lateral Anomaly Status x(i)'])
    s = int(df2['Affected Customers s(i)'])
    y1k = int(df2['~y(1)(ik)'])
    y2k = int(df2['~y(2)(ik)'])
    y3k = int(df2['~y(3)(ik)'])
    y4k = int(df2['~y(4)(ik)'])
    y1 = int(df2['~y(1)(i)'])
    y2 = int(df2['~y(2)(i)'])
    y3 = int(df2['~y(3)(i)'])
    y4 = int(df2['~y(4)(i)'])
    Ni = int(df2['N(i)'])

    x_val = int(pyt.pre_x(y1,y2,y3,y4))
    xik_val = int(pyt.pre_xik(l,N,x,s,y1k,y2k,y3k,y4k,y1,y2,y3,y4))
    s_val = float(pyt.pre_s(y1,y2,y3,y4,Ni))
    sa = float(df2['Affected Customers s(i)'])
    xa = int(df2['Lateral Anomaly Status x(i)'])
    xia = int(df2['Customer Anomaly Status x(i)(k)'])

    return render_template("sub.html", x = x_val, s1 = "{:.9f}".format(s_val), xi = xik_val, c = cust, l = late, s2 = "{:.9f}".format(sa), xa = xa, xia = xia)


if __name__=="__main__":
  app.run(debug=True)