# Disease-Modeling
Capstone Project for General Assembly

Projecting and Exploring U.S. Mortality from Major Diseases
N.B. Associated visualizations can be found both at Google Slides
(https://docs.google.com/presentation/d/1xuLCjb2GQULCUKoz5Yz79QyhbuNVj3Fg7uCVKO_vRhk/edit?usp=sharing) and at my
public Tableau (https://public.tableau.com/profile/greg2244#!/, then click on "Projecting U.S. Mortality from Major Diseases").

This project examines mortality rates from six major infectious diseases (tuberculosis, HIV/AIDS, lower respiratory disease,
meningitis, hepatitis (A), and diarrheal disease). The primary dataset is courtesy of the University of Washington's Institute
for Health Metrics and Evaluation and is available publicly at
http://ghdx.healthdata.org/record/united-states-infectious-disease-mortality-rates-county-1980-2014. Let's note here also that,
though all of these diseases are officially communicable, transmission between human beings often requires a particularly
intimate sort of contact. Contracting HIV, for example, requires absorption of an infected person's bodily fluids, while
hepatitis is often transmitted through fecal matter.

The IHME dataset comprises average mortality rates by sex for these six diseases between the years 1980 and 2014. My first goal
was to make projections about how rates would progress, and in particular to estimate which states would have the highest rates
for these diseases in 2019. To this end I used a gamma regression, implemented in statsmodels.api with its GLM (generalized
linear models) module. I simply took in the years and mortality rates from the dataset and then used the regression to predict
values for the rate on the next ten years (2015-2024). I constructed functions for this purpose so that I could quickly
calculate rates and predictions for all fifty states as well as the nation as a whole. So e.g. a typical function would look
like this:

```
def t_hiv(X, y):

    results = sm.GLM(y, sm.add_constant(X),
                sm.families.Gamma(sm.families.links.log)).fit()     # Log-fit with constant
                                                                    # term of Gamma Regression
    
    X_test = pd.DataFrame([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
                      columns = ['future'])                         # Years to predict on
    preds = []
    for i in range(10):
        preds.append(y[209] * np.exp(results.params[1]) ** (i + 1)) # The prediction for year
                                                                    # n + 1 is a simple matter
                                                                    # multiplying the
                                                                    # prediction for year n by
                                                                    # another factor of Euler's
                                                                    # number to our beta. The
                                                                    # index on the y term is
                                                                    # determined by the index
                                                                    # for the most recent
                                                                    # (2014) mortality rate of
                                                                    # the disease in question
                                                                    # in the data frame. For
                                                                    # HIV, it was 209.
                                        
    print(results.params[1])
    return pd.concat([X_test, pd.DataFrame(preds)], axis = 1)
```

I applied such a function to all fifty states' data frames, then extracted the 2019 row from each and threw these into a new
data frame. So for example the Wyoming contribution to the tuberculosis data frame was extracted via:

```
wy_tuber_df = t_tuberpred(wy.loc[(wy.index > 69) & (wy.index < 105), :]['year_id'],
                      wy.loc[(wy.index > 69) & (wy.index < 105), :]['mx'])    # 'wy' was the
                                                                              # data frame for
                                                                              # Wyoming. The
                                                                              # data for TB for
                                                                              # for both sexes
                                                                              # were between
                                                                              # row 70 and
                                                                              # 104.

wy_tuber = wy_tuber_df.loc[wy_tuber_df['future'] == 2019]

wy_tuber['state'] = 'Wyoming'

wy_tuber['new_index'] = 49

wy_tuber.set_index('new_index', inplace = True)
```

The results concerning the highest predictions for 2019 were as follows:

Tuberculosis: Alaska

HIV/AIDS: Florida

Lower respiratory disease: Mississippi

Meningitis: Louisiana

Hepatitis: California

Diarrheal disease: Rhode Island

The next goal was to try to understand what was responsible for these unusually high (predicted) rates.

Unsurprisingly, each case has its own idiosyncrasies, but a few general remarks are also in order. First, national rates for
all diseases are on a downward trend, with the exception of diarrheal disease. Tuberculosis and meningitis rates have been on
steady and (more or less) monotonic declines since 1980, but the graphs for the rest are different. HIV/AIDS had a significant
rise in the 1980s and a spike around 1993, while hepatitis had a significant rise in the 1990s and a spike around 2000.

In each case I brought in more data particular to the state in question in an attempt to gain some clarity about why it might
have such a high rate of the disease in question. I brought in climate data, population, and economic data, all split up by
county (or equivalent region).

The population and economic data were brought in using the BeautifulSoup webscraping tool. There were a few challenges getting
these new data together into the same data frame as the original IHME data, since counties were not always identically named.
Because there was only one row for each county in the new data, as opposed to 105 for each county in the IHME frame
(35 years * 3 genders (male, female, both)), I used an outer join to merge the two frames. For example, the case of Alaska
looked like this:

```
alaska_monies = pd.merge(alaska, moniesdf, how = 'left', left_on = 'location_name',
                        right_on = 'Borough or Census Area')
```

The climate data were entered manually based off of Köppen climate maps of the relevant states. So e.g. the Alaska climate map
looked like this: https://en.wikipedia.org/wiki/Climate_of_Alaska#/media/File:Alaska_K%C3%B6ppen.svg, and the resulting
dictionary to fill in climate values in the Alaskan data frame looked like this:

```
# The following dictionary maps counties to their Köppen climatic types. Obviously,
# the zones don't follow county lines with any precision. So for now I'll use slashes
# ('/') to separate names when a particular county features multiple climates.

ak_clim_dict = {'Aleutians West Census Area': 'Oceanic',
                'Aleutians East Borough': 'Tundra/Oceanic/Subarctic',
                'Kodiak Island Borough': 'Oceanic/Subarctic',
                'Sitka City and Borough': 'Oceanic',
                'Prince of Wales-Hyder Census Area': 'Oceanic',
                'Ketchikan Gateway Borough': 'Oceanic',
                'Kenai Peninsula Borough': 'Tundra/Dry-summer subarctic/Subarctic',
                'Nome Census Area': 'Tundra/Subarctic',
                'Bethel Census Area': 'Tundra/Dry-summer subarctic/Subarctic',
                'North Slope Borough': 'Tundra/Cold semi-arid/Dry-summer subarctic/Subarctic',
               'Lake and Peninsula Borough': 'Tundra/Subarctic',
               'Juneau City and Borough': 'Tundra/Subarctic',
                'Skagway Municipality': 'Dry-summer subarctic/Subarctic',
               'Haines Borough': 'Dry-summer subarctic/Subarctic',
               'Yakutat City and Borough': 'Tundra/Subarctic',
                'Valdez-Cordova Census Area': 'Tundra/Dry-summer subarctic/Subarctic',
               'Matanuska-Susitna Borough': 'Tundra/Dry-summer subarctic/Subarctic',
               'Southeast Fairbanks Census Area': 'Cold semi-arid/Subarctic',
               'Yukon-Koyukuk Census Area': 'Cold semi-arid/Dry-summer subarctic/Subarctic'}
```

Modeling
Because the population and economic data were largely from 2010, my general strategy was to exclude the observations from
before 2006 in constructing my models. In all cases I experimented with linear regressions and random forests, which were
constructed on training sets and scored on test sets. Here I made use of scikit-learn's 'linear_model' and 'ensemble' modules.
In general these performed quite well, though some of these models may be overfit. I also constructed simple feed-forward
neural nets using the keras tool. In general these were not many layers deep; the guiding philosophy in construction was to
have an input layer of neurons numbering somewhere around half the number of counties in the relevant state, with successive
layers reducing that number by half or so. Neurons in all layers were given the Rectified Linear Unit (ReLU) activation
function, with the exception of the single output neuron, where no activation function was needed. The models were compiled
with a mean squared error loss function and the 'adam' optimizer.

Finally, the counties of each state were examined for spatial correlation on the simple metric of 1's for bordering counties
and 0's otherwise. Spatially correlated data were discovered in the cases of Alaska (vis-à-vis tuberculosis) and Mississippi
(vis-à-vis lower respiratory diseases), with a marginal figure for California (vis-à-vis hepatitis). The Moran I-statistic (and
border matrix) was calculated by hand in the case of Rhode Island and in the case of Alaska, since the numbers of counties were
relatively small. In the other cases use was made of a .csv file (https://www.census.gov/geo/reference/county-adjacency.html)
that lists all the bordering counties for every county in the nation. Moran's I-statistic was calculated according to:

I = N / W * ΣiΣjwij(xi−x¯)(xj−x¯) / Σi(xi−x¯)^2,

where N is the number of regions (counties), W is the sum of the entries (0's for non-borderers, 1's for borderers) in the
border matrix, wij is the entry corresponding to county i and county j (the matrix is symmetric), and xi is the
mortality ratein county i.

Alaskan Tuberculosis
Alaskan tuberculosis shows some spatial correlation among its counties. In addition, there is a negative correlation between
the mortality rate and income level. The Kusilvak Census Area and, to a lesser extent, the Bethel Census Area, stand out.

Kusilvak and Bethel are in the remote and relatively poor west of Alaska. There are also very large percentages of native
peoples in these areas. Kusilvak is roughly 93% Native American
(https://en.wikipedia.org/wiki/Kusilvak_Census_Area,_Alaska#Demographics), while Bethel is roughly 82% Native American
(https://en.wikipedia.org/wiki/Bethel_Census_Area,_Alaska#Demographics). Kusilvak is last in the state in per capita income.

There is a corresponding lack of infrastructure in these areas as well. Indeed, these western areas of the state are completely
inaccessible by road (http://www.alaska-map.org/road-map.htm).

When we put all of these factors together, we begin to understand why the Alaskan mortality rate from tubercuosis is so high.
It is also worth noting here that TB used to be the #1 killer in Alaska in the earlier part of the twentieth century.

The underlying factor seems to be access to modern medicine and modern information about the disease. But this factor shows its
head, in our case, in the form of location and economic data.

Floridian HIV
The most interesting phenomenon here is Union County, which has some of the highest mortality rates from several diseases in
the nation. As it happens, Union County is home to a very large prison and hospital, which is the major factor in accounting
for the high HIV mortality rate. The county itself has only around 15000 people, while the prison houses over 2000 inmates
(https://en.wikipedia.org/wiki/Union_Correctional_Institution). HIV rates in general are much higher in prison populations
(https://www.cdc.gov/hiv/group/correctional.html).

Mississippian Lower Respiratory Disease
Lower Respiratory Disease is of course a major killer worldwide. What stands out in Mississippi is that there is a great
disparity between the rate for men and the rate for women, with the mortality rate for men being significantly higher. A recent
report (https://msdh.ms.gov/msdhsite/_static/resources/4775.pdf) shows that Mississippi has also unusually high rates of
obesity, smoking, and cardiovascular disease. In fact this last is the highest in the nation. Smoking is more common among men
than women in every state except Wyoming
(https://www.kff.org/other/state-indicator/smoking-adults-by-gender/?currentTimeframe=0&sortModel=%7B%22colId%22:%22Location%22,%22sort%22:%22asc%22%7D).

There is also a mild spatial relationship among Mississippi's counties vis-à-vis lower respiratory disease. Very generally,
rates are higher in the south and lower in the north. Speculation: Because the university towns of Oxford and Starkville are in
the northern part of the state, there may be more awareness of connections between diet/smoking and respiratory disease. By
contrast, residents in the gulf region of the state may have more influence from New Orleans as regards lifestyle.

Louisianan Meningitis
This one remains the most obscure to me. The only major correlation I can find is a negative one with year, which means only
that mortality rates from meningitis have dropped significantly in Louisiana over the last forty years. Why exactly it is
Louisiana that has the highest rates among all states I cannot ascertain. There is no correlation between mortality rate and
any climatic or population or economic variable that I have examined. It seems to affect men more than women, and possibly
other races more than whites, but there is nothing very significant here. Meningitis affects young children more than other
ages. The U.S. as a whole has a population that is 6.3% children under 5. Louisiana's rate is higher than average (6.7%), but
only slightly (https://www.indexmundi.com/facts/united-states/quick-facts/all-states/percent-of-population-under-5#map).
Louisiana's infant mortality rate is the fourth-highest in the nation
(https://www.cdc.gov/nchs/pressroom/sosmap/infant_mortality_rates/infant_mortality.htm), but this seems more like a
re-statement of our puzzle than a solution to it.

There are a variety of versions of meningitis; it can be caused by bacteria or viruses or fungi. HIV is a significant risk
factor for the fungal variety (https://en.wikipedia.org/wiki/Meningitis), which is possibly relevant here, as the HIV rate in
Louisiana is also relatively high.

Californian Hepatitis
Sex is a significant factor here, with men having significantly higher rates than women. Hepatitis A is often spread through
fecal contact, so issues like sanitation and homelessness are important considerations here. San Diego recently had an
outbreak, and San Francisco County is over-represented in our dataset.

Climate is mildly relevant here: The cold-summer Mediterranean climate is negatively correlated with hepatitis mortality; this
climate comprises the northern part of the state that's outside of the central valley.

Rhode Islander Diarrheal Disease
The main factor here is a positive correlation with year, as rates have been going up sharply. Climate and county are
marginally relevant, but the major point that calls our for explanation is the rising rate. My speculative hypothesis here is
that this is due to sharply rising rates of antibiotic and opioid use, which often have diarrhea as side effects. Diarrhea can
lead to fatal results through extreme dehydration, especially in young children, who tend to have more vulnerable organs.

Conclusions and Future Work
Conclusions
If there be one general lesson learned, it is that explanations for mortality rates are highly complex and idiosyncratic. In
the course of my exploration, very often I would lift up a rock and discover something interesting about one of the six
diseases or about some county's demographics, only to realize that the newly won knowledge simply raised more questions. It is
meet to bear in mind that there are often many layers of explanation, and how deep one wants or needs to go is generally a
contextual matter. Here are some facts that I think our work has explained, some more deeply than others:

Tuberculosis in Alaska:
Because tuberculosis has a pedigree of high mortality in Alaska, today's remote areas in the region with low-income levels and
high native populations are likelier to have higher mortality rates from TB than other areas. Tuberculosis is still the #1
cause of human death from infectious disease (https://www.nature.com/articles/nrmicro.2018.8).

HIV in Florida:
A state's average mortality rate, when that average is taken over counties (so that all counties, as opposed to all deaths, are
measured equally), can be significantly affected by small populations with unusually high mortality rates. Union County is such
a county because of its high prison population.

Lower Respiratory Diseases in Mississippi:
The connection between certain lifestyle patterns–most especially, smoking–and respiratory disease is well documented, but not
universally accepted. Mississippi has a high rates for smoking and poor diet, and there is evidence that the southern part of
the state has higher mortality rates for LRD than the northern part. This may be the result of a better educated populace (note
the locations of the University of Mississippi (Oxford) and Mississippi State University (Starkville)) in the north relative to
the south.

Meningitis in Louisiana:
Meningitis mortality rates in Louisiana have declined dramatically over the last forty years, but rates are still high.
Relatively high rates of HIV (which is linked with fungal meningitis) and relatively large numbers of young children (who are
most susceptible to the disease) are likely part of the full explanation.

Hepatitis in California:
Because hepatitis is often transmitted through fecal matter, hepatitis can be a major killer in areas of great urbanization and
poor sanitation. Thus e.g. there are many millions of sufferers in southern and southeastern Asia
(http://www.searo.who.int/india/topics/hepatitis/en/). Dense urban areas in California, like San Francisco and, more recently,
San Diego, seem especially prone to outbreaks. Mortality rates are lower in the cooler climatic zones.

Diarrheal Diseases in Rhode Island:
Deaths from diarrhea have been on a significant incline nationally over the last thirty years. We have speculated here that
increased use of antibiotics and opioids, especially by the elderly, have contributed to this rise.

Future Work
Perhaps it goes without saying that there is far more that could be examined using this dataset. I focused only on the states
with the highest predictions in 2019 for mortality rates of the six diseases. One could equally well be interested in the
states with the lowest rates and in what accounts for those. Moreover, because of the peculiar combinations of climatic,
economic, and demographic factors that characterize each state–and, indeed, each county–there is surely more to be learned
about why rates are the way that they are. The idiosyncrasies of Kusilvak Census Area, for example, may go a long way toward
explaining the high rates of tuberculosis in Alaska, but there are other states with high rates of tuberculosis mortality for
which the explanation must surely proceed along other lines.

Furthermore, whatever stories I have succeeded in uncovering are themselves incomplete. The national spike (though modest) in
hepatitis around 2000 demands more explanation, as does why Louisiana leads the nation (or likely will) in meningitis
mortality. Climatic considerations are largely absent from my explanations, but perhaps there is more to be said there as well.

Another point to be noted is that monetary factors are likely not ultimate causes. Though there is indeed a negative
correlation between e.g. median county income in Louisiana and mortality rate from meningitis, there is no reason to suppose
that poverty has some direct influence on the likelihood of contracting meningitis. The much more likely explanation is that
income is correlated with meningitis mortality through other factors such as education and access to medical resources.
