'''
Author(s):          Sam Badger

changelog:  03/28/2019  Initial results
            04/02/2019  Expand time window. accumulate daily returns
            04/05/2019  Use S&P500 as baseline. Calculate regression lines.
            04/13/2019  Fix regression lines and calculate residuals
'''

import csv

import copy
from datetime import datetime, timedelta

from scipy import stats
import numpy as np

from matplotlib.pyplot import subplot
import matplotlib.pyplot as plt

from math import sqrt


##############################################
# Global variables
##############################################
DAYS_BEFORE_EVENT = 30
DAYS_AFTER_EVENT = 30

##############################################
#   Time Series Analysis Functions
##############################################
def plotCompaniesTimeSeriesResiduals(companies, eventDate):
    # once all of the regression lines have been calculated, we can show the
    # residual plots and the r-squared values
    p=0
    for company in companies:
        p += 1
        com = companies[company]
        resid = calculateCompanyResiduals(com, eventDate)

        ax = subplot(5,1,p)
        ax.scatter(com['dates'], resid, label="Residuals")
        ax.set_xlim(com['dates'][0], com['dates'][-1])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper right')
        ax.axhline(0)
        ax.axvline(eventDate)
        ax.set_title("{} Residuals".format(company))
    plt.show()

def calculateCompanyResiduals(company, eventDate):

    def calcResiduals(fit, dates, vals):
        alpha = fit['intercept']
        beta = fit['slope']
        resid = []
        for i in range(len(dates)):
            x = (dates[i] - dates[0]).days
            y = vals[i]
            resid.append(y - (beta*x + alpha))
        return resid

    dates = np.array(company['dates'])

    before_regression = company['regression line']['before']
    before_dates = dates[dates<=eventDate]
    before_vals = company['adjusted cumulative returns'][:len(before_dates)]
    before_residuals = calcResiduals(before_regression, before_dates, before_vals)

    after_regression = company['regression line']['after']
    after_dates = dates[dates>=eventDate]
    after_vals = company['adjusted cumulative returns'][len(before_dates)-1:]
    after_residuals = calcResiduals(after_regression, after_dates, after_vals)

    residuals = before_residuals
    residuals.extend(after_residuals[1:])
    company['regression line']['residuals'] = residuals
    return company['regression line']['residuals']

def plotCompaniesTimeSeriesFits(companies, eventDate):

    p = 0
    for company in companies:
        p += 1

        dates = companies[company]['dates']
        acr = companies[company]['adjusted cumulative returns']

        # plot the cumulative returns
        ax = subplot(5,1,p)
        ax.plot(dates, acr)

        # mark the event date with a vertical line
        ax.axvline(eventDate)

        # retrieve the regression information
        regression = companies[company]['regression line']

        # draw the before time series regression lines
        alpha = regression['before']['intercept']
        beta = regression['before']['slope']
        before_x = [dates[0],eventDate]
        before_y = [alpha, alpha + beta * (eventDate-dates[0]).days]
        ax.plot(before_x, before_y, label = "Regression Before Event")

        # draw the after time series regression lines
        alpha = regression['after']['intercept']
        beta = regression['after']['slope']
        before_x = [eventDate,dates[-1]]
        before_y = [alpha, alpha + beta * (eventDate-dates[0]).days]
        ax.plot(before_x, before_y, label = "Regression After Event")

        # show the legends
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper right')
        ax.set_title("{} Hinged Regression".format(company))

    plt.show()

def fitCompaniesTimeSeries(companies, eventDate):
    for company in companies:

        com = companies[company]
        com['regression line'] = {}

        # call timeSeriesResgression
        dates = np.array(com['dates'])
        acr = com['adjusted cumulative returns']

        before_dates = dates[dates <= eventDate]
        before_acr = acr[0:len(before_dates)]
        fitCompanyPartialRegression(com, before_dates, before_acr, 'before');

        after_dates = dates[dates >= eventDate]
        after_acr = acr[len(before_dates)-1:]
        fitCompanyPartialRegression(com, after_dates, after_acr, 'after');

def fitCompanyPartialRegression(company, dates, values, part):
    slope, intercept, r_value, p_value, std_err = timeSeriesRegression(dates, values)

    company['regression line'][part] = {}
    regression = company['regression line'][part]
    regression['slope'] = slope
    regression['intercept'] = intercept
    regression['r squared'] = r_value ** 2
    regression['p value'] = p_value
    regression['std error'] = std_err

def timeSeriesRegression(dates, values):
    # convert the dates to integer values that are the number of days
    # since the first date.
    x = np.array([(date - dates[0]).days for date in dates])
    y = np.array(values)

    return stats.linregress(x,y)

##############################################
#   Time Independent Analysis Functions
##############################################
def getEventIndex(dates, eventDate):
    for i in range(len(dates)):
        if dates[i] == eventDate:
            return i
    raise Exception("Event date not found")

def plotTimeIndependentComparison(baseCompany, otherCompanies, eventDate):
    def abline(slope, intercept, color):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, color+'--')

    idx = getEventIndex(baseCompany['dates'], eventDate)

    p = 0
    for otherCompany in otherCompanies:
        p += 1

        x = baseCompany['adjusted cumulative returns']
        y = otherCompanies[otherCompany]['adjusted cumulative returns']

        # plot the cumulative returns
        ax = subplot(4,1,p)

        before_x = x[:idx+1]
        before_y = y[:idx+1]
        ax.scatter(before_x, before_y, c='r')

        after_x = x[idx:]
        after_y = y[idx:]
        ax.scatter(after_x, after_y, c='b')

        # retrieve the regression information
        regression = otherCompanies[otherCompany]['compare']['regression line']['before']
        abline(regression['slope'], regression['intercept'], 'r')
        regression = otherCompanies[otherCompany]['compare']['regression line']['after']
        abline(regression['slope'], regression['intercept'], 'b')

        ax.set_title("{} vs. NVIDIA Adjusted Cumulative Returns".format(otherCompany))

    plt.show()

def timeIndependentPartialRegression(com, baseComVals, otherComVals, part):
    com["compare"]['regression line'][part] = {}
    regression = com["compare"]['regression line'][part]

    slope, intercept, r_value, p_value, std_err = timeIndependentRegression(baseComVals, otherComVals)

    regression['slope'] = slope
    regression['intercept'] = intercept
    regression['r squared'] = r_value ** 2
    regression['p value'] = p_value
    regression['std error'] = std_err

def timeIndependentRegression(baseComVals, otherComVals):
    return stats.linregress(np.array(baseComVals), np.array(otherComVals))

def companyCompare(baseCom, otherCom, eventDate):
    otherCom["compare"] = {}
    otherCom["compare"]["regression line"] = {}

    assert getEventIndex(baseCom['dates'], eventDate) == getEventIndex(otherCom['dates'], eventDate)

    idx = getEventIndex(baseCom['dates'], eventDate)
    base_values_before = baseCom['adjusted cumulative returns'][:idx+1]
    other_values_before = otherCom['adjusted cumulative returns'][:idx+1]
    timeIndependentPartialRegression(otherCom, base_values_before, other_values_before, 'before')

    base_values_after = baseCom['adjusted cumulative returns'][idx:]
    other_values_after = otherCom['adjusted cumulative returns'][idx:]
    timeIndependentPartialRegression(otherCom, base_values_after, other_values_after, 'after')

def timeIndependentComparativeAnalysis(companies, eventDate):
    # separate NVIDIA data from other company data
    baseCompany = "NVIDIA CORP"
    nvidia = companies[baseCompany]

    other_companies = companies.copy()
    del other_companies[baseCompany]

    for company in other_companies:
        companyCompare(nvidia, other_companies[company], eventDate)

    plotTimeIndependentComparison(nvidia, other_companies, eventDate)


##############################################
#   Main Event Analysis Function
##############################################
def researchEvent(eventDate):
    dataDir = "../trimmed_data/"

    startDate = eventDate - timedelta(days=DAYS_BEFORE_EVENT)
    endDate = eventDate + timedelta(days=DAYS_AFTER_EVENT)
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

    ############################################################################
    # Calculating Regression Lines and Residuals
    ############################################################################
    # fit time-series regression for all companies and plot
    fitCompaniesTimeSeries(companies, eventDate)
    plotCompaniesTimeSeriesFits(companies, eventDate)
    plotCompaniesTimeSeriesResiduals(companies, eventDate)

    ############################################################################
    # Calculating Regression Lines and Residuals
    ############################################################################
    timeIndependentComparativeAnalysis(companies, eventDate)

if __name__ == "__main__":

    eventDates = [
        datetime(day=25, month=10, year=2016),  #GeForce GTX 1050 series
        datetime(day=17, month=5, year=2017)    #GeForce GT 1030
    ]

    for event in eventDates:
        researchEvent(event)
