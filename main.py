import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def show_data(df, x_values, y_values):
    if st.checkbox('Show all data'):
        st.write('All data')
        st.table(df)
    else:
        st.write('Top five rows of the data:')
        st.table(df.head())

    plot_scatter(df, x_values, y_values)

def plot_scatter(data_frame, x_values, y_values):
    plot = px.scatter(data_frame=data_frame, x=x_values, y=y_values)
    st.plotly_chart(plot)

def show_splitted_data(train_set, test_set):
    option = st.radio('Which data would you like to plot?', ('training set', 'test set'))
    st.write(f'Top five rows of {option}:')
    if option == 'training set':
        st.table(train_set.head())
        plot_scatter(train_set, train_set.columns[0], train_set.columns[1])
    else:
        st.table(test_set.head())
        plot_scatter(test_set, test_set.columns[0], test_set.columns[1])

def show_cost(Y_test, y_predict):
    option = st.radio('Choose the evaluation metric to see the cost of the model.', ('Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'))
            
    if option == 'Mean Absolute Error':
        cost = mean_absolute_error(Y_test, y_predict)
    elif option == 'Mean Squared Error':
        cost = mean_squared_error(Y_test, y_predict)
    else:
        cost = (np.sqrt(mean_squared_error(Y_test, y_predict)))

    st.write(f"The cost of the model using {option} is {round(cost, 3)}.")

def use_model(model, first_column_name, second_column_name, min_value, max_value):
    input = st.slider(first_column_name, min_value, max_value)
    x_predict = np.array([input])
    output = model.predict(x_predict.reshape(1,-1))
    st.write(f' If {first_column_name.lower()} is {input}, {second_column_name.lower()} is {round(float(output), 3)}.')

def main():
    ## Title and Intro
    _, col2, _ = st.columns(3)
    with col2:
        st.image("images/icon.png")

    st.markdown("<h2 style='text-align: center;'>Linear Regression Demo</h2>", unsafe_allow_html=True)
    st.markdown('''
        Upload any **csv** file containing **2 columns** containing numbers. Linear regression will be used to learn
        a relationship between the two columns and output a model eventually. This model can be used to predict **the value 
        of the second column if given a value of the first column.**  
    ''')

    ## Upload Data
    st.header('Upload Data')
    uploaded_file = st.file_uploader('Please upload a csv file.', type='csv', help='The csv file must have exactly 2 columns, both of which contain numbers.')

    if uploaded_file:
        ## Explore Data
        st.header('Explore Data')
    
        input_df = pd.read_csv(uploaded_file)

        first_column_name = input_df.columns[0]
        second_column_name = input_df.columns[1]

        x_values = input_df[first_column_name]
        y_values = input_df[second_column_name]

        X = np.array(x_values).reshape(-1,1)
        Y = np.array(y_values).reshape(-1,1)
        
        # show data in table form and chart form
        show_data(input_df, x_values, y_values)

        ## Split Data
        st.header('Split Data')
        test_size = st.slider(
            label='Select the test size (in percentage %).', 
            min_value=10, 
            max_value=90, 
            value=20, 
            step=10, 
            help='Select how much percentage of the input data will be used as the test size. Of course, the remaining data will be used as the training set.')
        
        train_set, test_set = train_test_split(input_df, test_size= test_size / 100, random_state=5)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size / 100, random_state=5)


        # expander to see splitted data
        with st.expander('See how the data is splitted into training set and test set.'):
            show_splitted_data(train_set, test_set)

        ## Train Model
        st.header('Train model')
        st.markdown(f'''
                Your model has been trained using
                - the training set having the size of {100 - test_size}% of data
                - the test set having the size of {test_size}% of data
                - linear regression algorithm
            ''')

        model = LinearRegression()

        model.fit(X_train, Y_train)

        # plot the model
        prediction_space = np.linspace(min(X), max(X)).reshape(-1,1) 
        plt.scatter(X_train,Y_train)
        plt.plot(prediction_space, model.predict(prediction_space), color = 'black', linewidth = 3)
        plt.xlabel(first_column_name) 
        plt.ylabel(second_column_name) 
        plt.savefig("images/output.png")
        st.image('images/output.png', width=500)

        ## Test Model
        st.title('Test Model')
        y_predict = model.predict(X_test)

        st.write('We have tested the model with the test set you provided.')
        # expander to see the cost of the model
        show_cost(Y_test, y_predict)

        ## Use Model
        st.header('Use Model')
        use_model(model, first_column_name, second_column_name, int(min(X)), int(max(X)))
            
if __name__ == '__main__':
    PAGE_CONFIG = {"page_title":"Linear Regression","page_icon":"images/icon.png","layout":"centered"}
    st.set_page_config(**PAGE_CONFIG)
    main()