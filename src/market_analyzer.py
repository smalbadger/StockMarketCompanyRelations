'''
Author(s):          Sam Badger

changelog:  03/28/2019  Initial results
            04/02/2019  Expand time window. accumulate daily returns
            04/05/2019  Use S&P500 as baseline. Calculate regression lines.
            04/13/2019  Fix regression lines and calculate residuals
'''

import csv
from datetime import datetime, timedelta
from matplotlib.pyplot import subplot
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from pandas_datareader import data
import pandas as pd
from pprint import pprint

from math import sqrt

def timeSeriesRegression(dates, values):
    # convert the dates to integer values that are the number of days 
    # since the first date.
    x = np.array([(date - dates[0]).days for date in dates])
    y = np.array(values)
    
    return stats.linregress(x,y)
    
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    
    cov = np.cov(x, y)
    var_x = 
    # calculate the covariance of time and adjusted cumulative returns and the 
    # variance of time. These values are used to calculate regression lines.
    var_acr=0
    for i in range(len(dates)):
        date = dates[i]
        cr = acr[i]
        
        var_acr += (cr-macr)**2
        var_time += ((date - mean_date).days) ** 2
        cov += (date - mean_date).days*(cr - macr)
    var_time /= (len(dates)-1)
    cov /= (len(dates)-1)
    var_acr /= (len(dates)-1)
    
    r = cov / (sqrt(var_time)*sqrt(var_acr))
    
    # calculate the slope and y-intercept of the regression line
    beta = cov/var_time
    alpha = macr - beta * (mean_date-startDate).days
    companies[company]['regression line'] = {}
    companies[company]['regression line']['alpha'] = alpha
    companies[company]['regression line']['beta'] = beta
    companies[company]['regression line']['r squared'] = r**2


def researchEvent(eventDate):
    dataDir = "../trimmed_data/"


    startDate = eventDate - timedelta(days=11)
    endDate = eventDate + timedelta(days=11)
    print("window: {}".format(endDate - startDate))
    
    ################################################################################
    #                           READ S&P500 DATA
    # We use the SPY ETF so we don't have to pull data from 500 companies.
    ################################################################################
    filename = "SPY.csv"

    spy = {"dates":[], "returns":[]}
    # first iteration calculates the returns
    with open(dataDir+filename) as f:
        dataReader = csv.DictReader(f)
        for row in dataReader:
            date = datetime.strptime(row["Date"],"%Y-%m-%d")
            price = float(row["Close"])
            
            if date > datetime.strptime("2016-01-04", "%Y-%m-%d"):
                ret = ((price - previous_price)/previous_price)
                spy['returns'].append(ret)
                spy['dates'].append(date)
                
            previous_price = price
        
    startIndex, endIndex = None, None
    for i in range(len(spy["dates"])):
        date = spy["dates"][i]
        if date <= startDate:
            startIndex = i
        if date <= endDate:
            endIndex = i
            
    # trim S&P 500 data to only the data that we want.
    for field in spy:
        spy[field] = spy[field][startIndex:endIndex+1]
        
    # accumulate the returns
    spy["cumulative returns"] = [1]
    for i in range(1,len(spy['returns'])):
        #spy["cumulative returns"].append(spy["cumulative returns"][-1] + spy['returns'][i])
        
        current = spy["returns"][i]
        previous_cumulative = spy['cumulative returns'][i-1]
        current_cumulative = (current + 1) * previous_cumulative
        spy["cumulative returns"].append(current_cumulative)
        spy["cumulative returns"][i-1] -= 1
        
    spy["cumulative returns"][-1] -=1

    ################################################################################
    #                           READ ALL COMPANY DATA
    # The company data was pulled from crisper by Billy. Since the files are so big,
    # the data was split into 4 different files. The data consists of returns for a 
    # 2 year period for over 6,000 companies.
    ################################################################################
    filenames = ["crsp16_1.csv","crsp16_2.csv","crsp17_1.csv","crsp17_2.csv"]

    # iterate through the data files, open them, and wrap them in DictReaders.
    dataReaders = [csv.DictReader(open(dataDir+name)) for name in filenames]

    # dictionary to store company data: companyName (str) -> data (dict)
    companies = {}

    # Companies of Interest
    COI = [
        "NVIDIA CORP",
        "APPLE INC",
        "ADVANCED MICRO DEVICES INC",
        "TAIWAN SEMICONDUCTOR MFG CO LTD",
        "INTEL CORP"
    ]

    ##################
    # parse the data #
    ##################
    for reader in dataReaders:
        for row in reader:
            company = row['COMNAM'].strip()

            # only keep companies we're interested in
            if company not in COI:
                continue

            # try to parse the date. If it doesn't work, move on. if the date is
            # not in the desired range, move on.
            date = datetime.strptime(row["DATE"], '%m/%d/%Y')
            if date < startDate or date > endDate:
                continue

            ret = row['RET']
            count = row['count']

            if date != "" and ret != "":
                if company not in companies:
                    companies[company] = {"dates":[], 
                                          "returns":[], 
                                          "counts":[], 
                                          "cumulative returns":[1],
                                          "adjusted cumulative returns":[]}
                companies[company]['dates'].append(date)
                companies[company]['returns'].append(float(ret))
                companies[company]['counts'].append(count)

    ##########################
    # Accumulate the returns
    ##########################
    for company in companies:
        cr = companies[company]['cumulative returns']
        r = companies[company]['returns']
        for i in range(1,len(r)):
            current = r[i]
            previous_cumulative = cr[i-1]
            current_cumulative = (current + 1) * previous_cumulative
            cr.append(current_cumulative)
            cr[i-1] -= 1
        cr[-1] -= 1


    ################################################################################
    # Plotting Cumulative and Adjusted Cumulative Returns
    ################################################################################

    # Plot the cumulative returns of the companies and the S&P500.
    ax = subplot(2,1,1)
    ax.set_title("Time vs. Cumulative Returns")
    for company in COI:
        ax.plot(companies[company]['dates'], companies[company]["cumulative returns"], label=company)
    ax.plot(spy["dates"], spy['cumulative returns'], label="SPY")
    ax.axvline(eventDate)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    ax.set_xlim(startDate,endDate)

    # Subtract the S&P500 from the companies and plot again.
    ax = subplot(2,1,2)
    ax.set_title("Time vs. Adjusted Cumulative Returns (S&P500 subtracted)")
    for company in COI:
        cr = companies[company]['cumulative returns'][:]
        for i in range(len(cr)):
            companies[company]['adjusted cumulative returns'].append(cr[i] - spy['cumulative returns'][i])
        ax.plot(companies[company]['dates'], companies[company]['adjusted cumulative returns'], label=company)
    ax.axvline(eventDate)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    ax.set_xlim(startDate,endDate)
    plt.show()

    ################################################################################
    # Calculating Regression Lines and Residuals
    ################################################################################

    others = dict(COI)
    nvidia = others["NVIDIA CORP"]
    del nvidia
    
    for company in others:
        # calculate regression lines for the days before the event
        # calculate regression lines for the days after the event

    
    # to calculate regression lines, we first calculate the variance of the x axis,
    # which is time. We also calculate the covariance of time and cumulative returns
    p=0
    for company in COI:
        p+=1
        
        # call timeSeriesResgression
        
        # plot the regression lines and vertical lines for the event date
        ax = subplot(5,1,p)
        ax.plot(companies[company]['dates'], companies[company]['adjusted cumulative returns'], label="{}     R squared = {:.3f}".format(company,r**2))
        x = [dates[0],dates[-1]]
        y = [alpha, alpha + beta * (dates[-1]-dates[0]).days]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper right')
        ax.plot(x,y)
        ax.axvline(eventDate)
        if p==1:
            ax.set_title("Fitting Regression Lines to Adjusted Cumulative Return")

    plt.show()

    # once all of the regression lines have been calculated, we can show the
    # residual plots and the r-squared values
    p=0
    for company in COI:
        p+=1
        com = companies[company]
        alpha = com['regression line']['alpha']
        beta = com['regression line']['beta']
        r2 = com['regression line']['r squared']
        acr = com['adjusted cumulative returns']
        dates = com['dates']
        
        com['residuals'] = []
        for i in range(len(dates)):
            x = (dates[i] - startDate).days
            y = acr[i]
            com['residuals'].append(y - (beta*x + alpha))
        
        ax = subplot(5,1,p)
        ax.scatter(com['dates'], com['residuals'], label="{}     R squared = {:.3f}".format(company,r2))
        ax.set_xlim(startDate,endDate)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper right')
        ax.axhline(0)
        ax.axvline(eventDate)
        if p==1:
            ax.set_title("Residual Plots From Fitting Regression Lines to Adjusted Cumulative Return")
    plt.show()
 
if __name__ == "__main__":

    # NVIDIA releases the GeForce GTX 1050 series
    #researchEvent(datetime(day=25, month=10, year=2016))
    
    # NVIDIA releases the GeForce GT 1030
    researchEvent(datetime(day=17, month=5, year=2017))
    
