# Predicting Adoption of Solar Power Across the United States

There is no denying the impact that climate change has had on the planet. According to the Intergovernmental Panel on Climate Change (IPCC) in their Sixth Assessment Report, climate change is a result of human actions, including increases in CO<sub>2</sub> and greenhouse gas emissions, and it is already affecting weather and extreme events worldwide.<sup>1</sup> At this point, the world needs to limit emissions to prevent warming the planet to 2<sup>o</sup>C; a threshold where extreme events would become even more widespread. In April of 2021, President Biden committed to reduce the United States greenhouse gas emissions by 50-52% by 2030 as part of the country’s contribution to the Paris Climate Agreement.<sup>2</sup> The US emits the second most greenhouse gases in the world, so hitting these targets and even going beyond them would help to curb warming effect on the climate. Electricity generation in the US is mainly fueled by coal and natural gas, which is why this area accounts for 25% of the country’s greenhouse gas emissions. By aggressively switching to renewable energy sources of electricity, the country can begin to achieve the climate goals and prevent further warming of the planet. Solar is a likely candidate as a renewable energy source because the cost of solar has dropped more than 70% over the last decade and is now competitively priced with other dirtier fuel sources.<sup>3</sup>

This report aims to identify drivers of solar adoption in both the residential and the industrial sectors through a machine learning predictive model. For each sector, the model attempts to predict solar panel system count at the county level using a combination of socioeconomic, policy, and financial data related to the electricity market.

## Data
The DeepSolar group at Stanford University created a comprehensive dataset that identified the size, type, and number of solar panels in each county across the United States.<sup>4</sup> The solar panel data was determined through a convolutional neural network (CNN) that used satellite imagery to classify panels and estimate their sizes with precision and recall performance around 90%.5 The dataset includes information for all 48 contiguous U.S. states. There are 72,537 rows and 168 columns. These correspond with information for 72,537 twelve digit Federal Information Processing System (FIPS) codes for census tract locations across the U.S. The columns included in the dataset fall within five categories: 
1.	DeepSolar data\
a.	Data related to the CNN DeepSolar model, including variables such as total solar area, tile count, etc. for residential and non-residential panels\
b.	Source: DeepSolar model 
2.	U.S. Census data\
a.	Data related to the U.S. Census, including variables such as education levels, diversity, income, etc.\
b.	Source: ACS 2015
3.	Electricity data\
a.	Data related to the electricity such as consumption and prices\
b.	Source: U.S. Energy Information Administration
4.	Weather data\
a.	Data related to weather, such as solar radiation, temperature, elevation, etc.\
b.	Source: NASA Surface Meteorology and Solar Energy
5.	Political data\
a.	Data related to political leaning of the area, including variables such as democratic and republican voting percentages in 2012 and 2016\
b.	Source: Townhall.com and theguardian.com
6.	Policy data\
a.	Data related to incentives, taxes, and tariffs around solar energy\
b.	Source: NC State Clean Energy Technology Center

## Results
XGBoost performed the best for both the Residential and Non-Residential models.

| Target Sector | Method | Architecture | Best MSE | R<sup>2</sup> | Top <br> Features|
| ------------- | ------ | ------------ | -------- | ------------- | ------- |
Residential | Random <br> Forest | n_estimators: 1000 <br> min_samples_split: 5 <br> min_samples_leaf: 5 <br> max_features: 'sqrt' <br> max_depth: 20 <br> bootstrap: False| 552 | 0.753 | Education college <br> Daily solar radiation <br> Heating gas fuel | 
| Residential | XGBoost | n_estimators: 1000 <br> learning_rate: 0.01 | 503 | 0.769 | Incentive count residential <br> Daily solar radiation <br> Education college |
| Residential | Neural <br> Network | 3 hidden layers (128 units each, ReLU activation function) | 616 | 0.723 | Frost days <br> Daily solar radiation | Housing unit median value
| Non-<br>Residential | Random <br> Forest | Default | 49.6 | 0.249 | Total area <br> Population density <br> Relative humidity | 
| Non-<br>Residential | XGBoost | reg_lambda: 3 <br> reg_alpha: 1 <br> n_estimators: 300 <br> min_child_weight: 10 <br> max_depth: 20 <br> learning_rate: 0.01 <br> gamma: 0 <br> col_sample_bytree: 0.8 <br> booster: gbtree | 47 | 0.28 | Daily solar radiation <br> Industrial electricity price <br> Heating fuel solar | 
|Non-<br>Residential | Neural<br>Network | 3 hidden layers (128 units each, ReLU activation function) <br> 20% dropout between layers 2 and 3 | 50 | 0.227 | Population density <br> Frost days <br> Total area <br> Daily solar radiation

The below plots show the correlation between the predicted and actual values for the best performing models as well as the top features. 

### Residential Model Plots
<p align="center">
    <img src="https://github.com/gnyirjesy/Solar-Panel-Area-Prediction/blob/master/plots/xgboost_residential_system_count_errors_default.png?raw=TRUE">
    <img src = "https://github.com/gnyirjesy/Solar-Panel-Area-Prediction/blob/master/plots/xgboost_residential_fi.png?raw=TRUE">
</p>


### Non-Residential Model Plots
<p align="center">
    <img src="https://github.com/gnyirjesy/Solar-Panel-Area-Prediction/blob/master/plots/xgboost_nonresidential_scatter.png?raw=TRUE">
    <img src = "https://github.com/gnyirjesy/Solar-Panel-Area-Prediction/blob/master/plots/xgboost_nonresidential_fi.png?raw=TRUE">
</p>

## Future Work
The non-residential models tended to have low correlation between predicted values and actual values. In the future, this model could be improved by performing an initial classification to determine if a FIPS location has any non-residential solar systems before implementing the regression model on only those locations with non-residential systems. This additional step would help to address the class imbalance where most counties do not have any non-residential solar systems. \
Further improvements could also include implementing more extensive hyperparameter tuning using a grid search over all settings rather than a random sample. Grid search is a computational and time intensive process and therefore was not within the scope of this project.  

## Sources
1.	IPCC, 2021: Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change [Masson-Delmotte, V., P. Zhai, A. Pirani, S.L. Connors, C. Péan, S. Berger, N. Caud, Y. Chen, L. Goldfarb, M.I. Gomis, M. Huang, K. Leitzell, E. Lonnoy, J.B.R. Matthews, T.K. Maycock, T. Waterfield, O. Yelekçi, R. Yu, and B. Zhou (eds.)]. Cambridge University Press. In Press.
2.	Kidd, D. (2021, June 16). US regulatory barriers to an ambitious Paris Agreement Commitment - Environmental & Energy Law Program. Harvard Law School. Retrieved December 21, 2021, from https://eelp.law.harvard.edu/2021/04/us-paris-commitment/ 
3.	Solar Energy Industry Association. (2021). Solar Industry Research Data. SEIA. Retrieved December 21, 2021, from https://www.seia.org/solar-industry-research-data 
4.	The deepsolar project. Home - DeepSolar. (2018). Retrieved December 21, 2021, from http://web.stanford.edu/group/deepsolar/home 
5.	Yu, J., Wang, Z., Majumdar, A., & Rajagopal, R. (2018). DeepSolar: A machine learning framework to efficiently construct a solar deployment database in the United States. Joule, 2(12), 2605–2617. https://doi.org/10.1016/j.joule.2018.11.021 
6.	Singhal, S. (2021, June 21). Imputation techniques: What are the types of imputation techniques. Analytics Vidhya. Retrieved December 21, 2021, from https://www.analyticsvidhya.com/blog/2021/06/defining-analysing-and-implementing-imputation-techniques/ 
7.	Toloşi, L., & Lengauer, T. (2011). Classification with correlated features: Unreliability of feature ranking and solutions. Bioinformatics, 27(14), 1986–1994. https://doi.org/10.1093/bioinformatics/btr300 
8.	Pedregosa, F. a. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
9.	Kjytay. (2021, February 19). The box-cox and Yeo-Johnson transformations for continuous variables. Statistical Odds & Ends. Retrieved December 21, 2021, from https://statisticaloddsandends.wordpress.com/2021/02/19/the-box-cox-and-yeo-johnson-transformations-for-continuous-variables/ 
10.	Breiman, L. (1999, September). Random forests - statistics at UC Berkeley. Retrieved December 21, 2021, from https://www.stat.berkeley.edu/~breiman/random-forests.pdf 
11.	Gentine, P. (n.d.). Regression Tree [PowerPoint presentation]. Retrieved from Columbia University Courseworks.
12.	Chen, T., & Guestrin, C. (2016). XGBoost. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/2939672.2939785 
13.	Pykes, K. (2021, December 14). Fighting overfitting with L1 or L2 regularization: Which one is better? neptune.ai. Retrieved December 21, 2021, from https://neptune.ai/blog/fighting-overfitting-with-l1-or-l2-regularization 
14.	Shapley, Lloyd S. “A value for n-person games.” Contributions to the Theory of Games 2.28 (1953): 307-317.
15.	Gentine, P. (n.d.). Neural Networks [PowerPoint presentation]. Retrieved from Columbia University Courseworks.
