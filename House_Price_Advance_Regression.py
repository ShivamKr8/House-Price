
# 1. IMPORT LIBRARIES
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1.2 Import Data
trainData= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
testData= pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

### Start Exploring Data
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
trainData.head() 

trainData.info()

trainData.describe() 

#Let's Clean Our Data Into 3 Steps
#1. Duplicate ? 
display(trainData.duplicated().sum())

#2. Nulls ? # There Are Many Nulls We Will Handle It Now
pd.set_option('display.max_rows',10)
display(trainData.isnull().sum())
#3. Outliers ? Check The End 

# Check Null % For The Rest Small Cols #(BsmtFinType1,BsmtFinType2,Electrical,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageQual & GarageCond)
#for BsmtQual
## All Of this Is Small % So We Will Replace It With Mean
print('BsmtQual Null Percentage Is :')
AllNull= trainData['BsmtQual'].isnull().sum()/ len(trainData['BsmtQual'])*100
print(round(AllNull, 2))
#for BsmtCond
print('BsmtCond Null Percentage Is :')
AllNull= trainData['BsmtCond'].isnull().sum()/ len(trainData['BsmtCond'])*100
print(round(AllNull, 2))
#for BsmtExposure
print('BsmtExposure Null Percentage Is :')
AllNull= trainData['BsmtExposure'].isnull().sum()/ len(trainData['BsmtExposure'])*100
print(round(AllNull, 2))
#for BsmtFinType1
print('BsmtFinType1 Null Percentage Is :')
AllNull= trainData['BsmtFinType1'].isnull().sum()/ len(trainData['BsmtFinType1'])*100
print(round(AllNull, 2))
#for BsmtFinType2
print('BsmtFinType2 Null Percentage Is :')
AllNull= trainData['BsmtFinType2'].isnull().sum()/ len(trainData['BsmtFinType2'])*100
print(round(AllNull, 2))
#for Electrical
print('Electrical Null Percentage Is :')
AllNull= trainData['Electrical'].isnull().sum()/ len(trainData['Electrical'])*100
print(round(AllNull, 2))
#for FireplaceQu
print('FireplaceQu Null Percentage Is :')
AllNull= trainData['FireplaceQu'].isnull().sum()/ len(trainData['FireplaceQu'])*100
print(round(AllNull, 2))
#for GarageType
print('GarageType Null Percentage Is :')
AllNull= trainData['GarageType'  ].isnull().sum()/ len(trainData['GarageType'])*100
print(round(AllNull, 2))
#for GarageYrBlt
print('GarageYrBlt Null Percentage Is :')
AllNull= trainData['GarageYrBlt'  ].isnull().sum()/ len(trainData['GarageYrBlt'])*100
print(round(AllNull, 2))

# we can complete It as a calculations Like above but what if we can get it as a graph! let's See

def plot_nas(trainData: pd.DataFrame):
    if trainData.isnull().sum().sum() != 0:
        na_df = (trainData.isnull().sum() / len(trainData)) * 100      
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %' :na_df})
        missing_data.plot(kind = "barh")
        plt.show()
    else:
        print('No NAs found')
plot_nas(trainData)
# So we Will Replace All Nulls with Mean Except Last 4 Cols

display(trainData['LotFrontage'].describe())
# So Top Is 70
trainData['LotFrontage'].fillna("70", inplace = True)
trainData['LotFrontage'].isnull().sum()

display(trainData['BsmtQual'].describe())
# So Top Is TA 
trainData['BsmtQual'].fillna("TA", inplace = True)
trainData['BsmtQual'].isnull().sum()

display(trainData['BsmtCond'].describe())
# So Top Is TA 
trainData['BsmtCond'].fillna("TA", inplace = True)
trainData['BsmtCond'].isnull().sum()

display(trainData['BsmtExposure'].describe())
# So Top Is No
trainData['BsmtExposure'].fillna("No", inplace = True)
trainData['BsmtExposure'].isnull().sum()

display(trainData['BsmtFinSF1'].describe())
# So Top Is 438.242279
trainData['BsmtFinSF1'].fillna("438.242279", inplace = True)
trainData['BsmtFinSF1'].isnull().sum()

display(trainData['BsmtFinType1'].describe())
# So Top Is Unf
trainData['BsmtFinType1'].fillna("Unf", inplace = True)
trainData['BsmtFinType1'].isnull().sum()

display(trainData['BsmtFinType2'].describe())
# So Top Is Unf
trainData['BsmtFinType2'].fillna("Unf", inplace = True)
trainData['BsmtFinType2'].isnull().sum()

display(trainData['Electrical'].describe())
# So Top Is SBrkr
trainData['Electrical'].fillna("SBrkr", inplace = True)
trainData['Electrical'].isnull().sum()

display(trainData['FireplaceQu'].describe())
# So Top Is Gd
trainData['FireplaceQu'].fillna("Gd", inplace = True)
trainData['FireplaceQu'].isnull().sum()

display(trainData['GarageType'].describe())
# So Top Is Attchd
trainData['GarageType'].fillna("Attchd", inplace = True)
trainData['GarageType'].isnull().sum()

display(trainData['GarageYrBlt'].describe())
# So Top Is 1978.460756
trainData['GarageYrBlt'].fillna("1978.460756", inplace = True)
trainData['GarageYrBlt'].isnull().sum()

display(trainData['GarageFinish'].describe())
# So Top Is Unf
trainData['GarageFinish'].fillna("Unf", inplace = True)
trainData['GarageFinish'].isnull().sum()

display(trainData['GarageQual'].describe())
# So Top Is TA
trainData['GarageQual'].fillna("TA", inplace = True)
trainData['GarageQual'].isnull().sum()

display(trainData['GarageCond'].describe())
# So Top Is TA
trainData['GarageCond'].fillna("TA", inplace = True)
trainData['GarageCond'].isnull().sum()

display(trainData['MasVnrType'].describe())
# So Top Is None
trainData['MasVnrType'].fillna("None", inplace = True)
trainData['MasVnrType'].isnull().sum()

display(trainData['MasVnrArea'].describe())
# So Top Is 102.543133
trainData['MasVnrArea'].fillna("102.543133", inplace = True)
trainData['MasVnrArea'].isnull().sum()

#Drop PoolQC
trainData= trainData.drop('PoolQC', axis=1) ;
#Drop MiscFeature
trainData= trainData.drop('MiscFeature', axis=1)
#Drop Fence
trainData= trainData.drop('Fence', axis=1) ; 
#Drop Alley
trainData= trainData.drop('Alley', axis=1)
trainData.isnull().sum()

def plot_nas(trainData: pd.DataFrame):
    if trainData.isnull().sum().sum() != 0:
        na_df = (trainData.isnull().sum() / len(trainData)) * 100      
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %' :na_df})
        missing_data.plot(kind = "barh")
        plt.show()
    else:
        print('No Nulls found')
plot_nas(trainData)
#Final Check For Any Nulls ? Okay Finally No Nulls

# we Need To Know Numerical Cols that may be have Outliers so we need this 
trainData.select_dtypes(include=np.number).columns.tolist()

#Just To Remember Our Data
trainData.head(10)

#Let's Go From Bottom To Top "Start With Price"
##First See Some Details Then check For Outliers 
display(pd.DataFrame(trainData['SalePrice'].describe()))
sns.displot(trainData['SalePrice'], kind='hist',
            fill = True, color = 'g');
# As We See Before Eliminate Outliers the mean is 180921.195890

#First Step : Get Calcs
Q1= trainData['SalePrice'].quantile(.25)
Q3= trainData['SalePrice'].quantile(.75)
IQR= Q3-Q1
lowLimit= Q1-1.5*IQR
highLimit= Q3+1.5*IQR
print("Lower Limit is :" , lowLimit)
print("Upper Limit is :" , highLimit)

sns.boxplot(data=trainData[["SalePrice"]], orient="h")
#Simple Graph Shows Us Amount Of It
pricebef= trainData['SalePrice'].mean()
print(pricebef)

#### we did this step to get mean without OUTLIERS ####
#Second Step : Eliminate The Outliers
#trainData = trainData.drop(trainData[trainData.SalePrice > highLimit].index)
#trainData = trainData.drop(trainData[trainData.SalePrice < lowLimit].index)
#Third Step : See Diffrence In Graphs & Mean
display(pd.DataFrame(trainData['SalePrice'].describe()))
sns.boxplot(data=trainData[["SalePrice"]], orient="h")
# As We See After Eliminate Outliers the mean is 170244.449928 After Wliminate OutLiers

rainData['SalePrice'] = np.where(trainData['SalePrice'] < lowLimit, 170244.449928, trainData['SalePrice'])
trainData['SalePrice'] = np.where(trainData['SalePrice'] > highLimit, 170244.449928, trainData['SalePrice'])
sns.boxplot(data=trainData[["SalePrice"]], orient="h")

priceaft= trainData['SalePrice'].mean()
print("The Diffrence In Price After Eliminate Outliers Is : " )
print(round(priceaft-pricebef))

