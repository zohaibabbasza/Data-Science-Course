from sklearn.linear_model import LinearRegression   

model = LinearRegression()
model.fit(x_values, y_values)   
print(model.predict([ [127], [248] ]))