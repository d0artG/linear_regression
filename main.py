import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")


def mean_squared_error(m,b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        total_error += (y - (m*x + b))**2

    return (total_error/float(len(points)))

def gradient_descent(b_current, m_current, rate, points):
    m_gradient = 0
    b_gradient = 0
    N=float(len(points))
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        m_gradient += -(2/N) * (x *(y-(m_current*x + b_current)))
        b_gradient += -(2/N) * (y- (m_current*x + b_current))

    m_new = m_current - rate * m_gradient
    b_new = b_current - rate * b_gradient

    return m_new, b_new

m = 0
b = 0
rate = 0.00001
iterations = 300

for i in range(iterations):
    m, b = gradient_descent(m,b,rate,data)

print(m,b)

plt.scatter(data.studytime, data.score, color="black")
plt.plot(list(range(20,80)), [m*x+b for x in range(20,80)], color="red")
plt.show()


