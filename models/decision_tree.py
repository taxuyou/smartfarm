import re
import math
import xgboost as xgb
from utils.plotutils import draw_dt_feature_importance

def get_decision_tree_model(args):
    model = xgb.XGBRegressor(objective='reg:squarederror', 
                             n_estimators=args.train.trees,
                             verbosity=1,
                             tree_method='gpu_hist')
    return model


def get_multiout_decision_tree_model(args):
    model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', 
                                       n_estimators=args.train.trees,
                                       verbosity=1,
                                       tree_method='gpu_hist'))

    return model


def save_feature_importance(model, dataframe, lookback, top=10):
    # get feature names and their f1 score (or called f score)
    # f1 score = 2 * (precision / (precision + recall))
    # feature_names = model.get_booster().feature_names
    feature_importance = model.get_booster().get_score(importance_type='weight')
    '''
    keys = list(feature_importance.keys())
    values = list(feature_importance.values())
    print(feature_names)
    print(feature_importance)
    print(keys)
    print(values)
    '''

    # get top-10 feature names and their importance
    high_importance = sorted(list(feature_importance.values()), reverse=True)
    top_scores = high_importance[:10]
    top_indexes = []

    for i in range(len(top_scores)):
        name = [key for key, value in feature_importance.items() if value == top_scores[i]]
        if len(name) > 1:
            name = sorted(name, reverse=False)
        for j in range(len(name)):
            index = list(map(int, re.findall(r'\d+', name[j])))[0] 
            if index not in top_indexes:
                top_indexes.append(index)
            
    print(top_scores)
    print(top_indexes)

    # show each feature's column name, importance, and week
    fea_num = len(dataframe.columns)
    exp_dir = 'experiments/rda_decision_tree'
    f = open(exp_dir+'/feature_importance.txt', 'w')
    f.write("score, week, feature\n")
    f.write("--------------------\n")

    for score, index in zip(top_scores, top_indexes):
        week = math.ceil(index/fea_num)
        if week == 0: 
            week += 1
        
        # debug
        assert 1 <= week <= lookback, 'wrong week'
   
        print("importance: {}, week: {}, feature: {}".format(score, week, dataframe.columns[index%fea_num]))
        f.write(str(score)+", "+str(week)+", "+dataframe.columns[index%fea_num]+"\n")
    
    f.close()