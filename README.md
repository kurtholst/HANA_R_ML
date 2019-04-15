# HANA_R_ML
SAP HANA Machine Learning from R


# Blog on SAP HANA and R using the HANA ML R package
# 02.04.2019

# Clean environment
rm(list=ls())

# Load libraries
library(hana.ml.r)     # HANA R Library

# Connection to HANA Server.
conn <- hanaml.ConnectionContext(dsn = 'hxe',
                                 username = 'zyx',
                                 password = 'xyz')

train <- conn$table(table = "INSURANCE_TRAIN",
                    schema = "xyzx")

# R -> SQL. What is happening behind the scenes.
train$select.statement

# Fetch data to local R client. Easy - however not nessesary with SAP HANA PAL.
train_local <- train$Collect(n=100)

# Checking structure
str(train)
str(train_local)

# Exploring data
sprintf("Number of rows in the train dataset: %s", train$nrows)


train$Head(n=3)  # HANA.ML.R package SQL Select top x
head(train_local, n=3) # Basic R.

train$describe() # Summary
summary(train_local) # Basic R.


# Preparing featurelist and label (the very simple way)
featurelist <- list("Customer_Subtype" , "Number_of_houses" , "Avg_size_household" , "Avg_age" , "Customer_main_type" , "Roman_catholic" , "Protestant" , "Other_religion" , "No_religion" , "Married" , "Living_together" , "Other_relation" , "Singles" , "Household_without_children" , "Household_with_children" , "High_level_education" , "Medium_level_education" , "Lower_level_education" , "High_status" , "Entrepreneur" , "Farmer" , "Middle_management" , "Skilled_labourers" , "Unskilled_labourers" , "Social_class_A" , "Social_class_B1" , "Social_class_B2" , "Social_class_C" , "Social_class_D" , "Rented_house" , "Home_owners" , "One_car" , "Two_cars" , "No_car" , "National_Health_Service" , "Private_health_insurance" , "Income_LT_30000" , "Income_30_45000" , "Income_45_75000" , "Income_75_122000" , "Income_GT123000" , "Average_income" , "Purchasing_power_class" , "Contribution_private_third_party_insurance" , "Contribution_third_party_insurance_firms" , "Contribution_third_party_insurane_agriculture" , "Contribution_car_policies" , "Contribution_delivery_van_policies" , "Contribution_motorcycle_scooter_policies" , "Contribution_lorry_policies" , "Contribution_trailer_policies" , "Contribution_tractor_policies" , "Contribution_agricultural_machines_policies" , "Contribution_moped_policies" , "Contribution_life_insurances" , "Contribution_private_accident_insurance_policies" , "Contribution_family_accidents_insurance_policies" , "Contribution_disability_insurance_policies" , "Contribution_fire_policies" , "Contribution_surfboard_policies" , "Contribution_boat_policies" , "Contribution_bicycle_policies" , "Contribution_property_insurance_policies" , "Contribution_social_security_insurance_policies" , "Number_of_private_third_party_insurance" , "Number_of_third_party_insurance_firms" , "Number_of_third_party_insurane_agriculture" , "Number_of_car_policies" , "Number_of_delivery_van_policies" , "Number_of_motorcycle_scooter_policies" , "Number_of_lorry_policies" , "Number_of_trailer_policies" , "Number_of_tractor_policies" , "Number_of_agricultural_machines_policies" , "Number_of_moped_policies" , "Number_of_life_insurances" , "Number_of_private_accident_insurance_policies" , "Number_of_family_accidents_insurance_policies" , "Number_of_disability_insurance_policies" , "Number_of_fire_policies" , "Number_of_surfboard_policies" , "Number_of_boat_policies" , "Number_of_bicycle_policies" , "Number_of_property_insurance_policies" , "Number_of_social_security_insurance_policies")
label <- "Number_of_mobile_home_policies_num"

# Ensemble Decision Tree using random forest
model <- hanaml.RandomForestClassifier(conn.context = conn,
                                       df = train, 
                                       max.depth = 5,
                                       features = featurelist,
                                       label = label)
str(model)


# Model evaluation
model$confusion.matrix_
oob <- model$oob.error$Collect()
sprintf("The average out of bag error, of all trees titted is: %s", mean(oob[[2]]))
plot(oob)

# Save trained model to SAP HANA
modeldf <- model$model$Collect()
model$model$save(where = 'MODELRF_SST1',  table.type = NA)

  
# Feature importance
model$feature_importances_$Collect()


# Predicting on new data
test <- conn$table(table = "INSURANCE_TEST",
                      schema = "xyzx")

predicted <- predict(model = model,
                     df = test,
                     key = "ID",
                     features = featurelist)

