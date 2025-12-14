print("Testing packages...")

try:
    import pandas
    print(" pandas")
except:
    print(" pandas FAILED")

try:
    import numpy
    print(" numpy")
except:
    print(" numpy FAILED")

try:
    import yfinance
    print("✅ yfinance")
except:
    print(" yfinance FAILED")

try:
    import ta
    print(" ta (technical analysis)")
except:
    print(" ta FAILED")

try:
    import sklearn
    print(" scikit-learn")
except:
    print(" scikit-learn (optional)")

try:
    import matplotlib
    print(" matplotlib")
except:
    print(" matplotlib (optional)")

try:
    import flask
    print(" flask")
except:
    print(" flask (optional)")

print("\n DONE! If pandas, numpy, yfinance, and ta are OK, you're ready!")