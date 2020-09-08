import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
import tensorflow as tf


__all__ = ['draw_heatmap', 'draw_bargraph', 'draw_harvest_per_sample',
           'draw_rda_dataframe', 'draw_dt_result_compare_1', 
           'draw_dt_result_compare_2', 'draw_dt_feature_importance',
           'draw_bl_losses', 'draw_bl_result_compare_1',
           'draw_bl_result_compare_2',
          ]

'''
plot one shot
'''
def draw_heatmap(heatmap, x_labels, y_labels, filename):

    plt.figure()
    res = sns.heatmap(data=heatmap, linewidth=0.5, xticklabels=x_labels, yticklabels=y_labels, cmap='YlGnBu')
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize=6)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize=6)
    plt.title(filename)
    plt.tight_layout()
    plt.savefig(filename)


def draw_bargraph(data, filename, x_labels=None, reverse_index=False):
    if reverse_index:
        index = [i for i in range(len(data), 0, -1)]
    else:
        index = np.arange(len(data))
    plt.figure()
    plt.bar(index, data)
    plt.title(filename)
    if x_labels is not None:
        plt.xticks(index, x_labels, rotation=90)
    plt.tight_layout()
    plt.savefig(filename)

def autolabel(ax, rects, xpos='center'):
    ha = {'center':'center', 'right':'left', 'left':'right'}
    offset = {'center':0, 'right':1, 'left':-1}

    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate('{0:.2f}'.format(height), 
                        xy=(rect.get_x()+rect.get_width()/2, height),
                        xytext=(offset[xpos]*3, 2),
                        textcoords="offset points",
                        ha=ha[xpos], va='bottom')

def draw_harvest_per_sample(out, gt, filename):
    b, h, w, c = out.shape
    
    h, w = int(h), int(w)
    width = 0.35
    index = np.arange(w)
    
    cols = 4
    rows = int(np.ceil(h / cols))
    
    fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(16, 16))
    out = tf.reshape(out, [h, w])
    gt = tf.reshape(gt, [h, w])
    
    r_index, c_index, s_index = 0, 0, 1
    for o, g in zip(out, gt):
        rects1 = axes[r_index][c_index].bar(index - width/2, o, width, label='Pred')
        rects2 = axes[r_index][c_index].bar(index + width/2, g, width, label='Truth')
        axes[r_index][c_index].set_title("Sample {}".format(s_index))
        axes[r_index][c_index].legend()
        autolabel(axes[r_index][c_index], rects1, "left")
        autolabel(axes[r_index][c_index], rects2, "right")
        
        c_index += 1
        s_index += 1
        
        if c_index == cols:
            r_index += 1
            c_index = 0
    
    d_index = r_index * cols + c_index
    
    for i in range(d_index, rows * cols):
        fig.delaxes(axes.flatten()[i])
        
    fig.tight_layout()
    plt.savefig(filename)


'''
plot one sheet
'''
def draw_rda_dataframe(dataframe, groups, sheet, size=5):
    values = dataframe.values
    i = 1
    plt.figure(figsize=(size,size))
    
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(values[:, group])
        plt.title(dataframe.columns[group], y=0.5, loc='right')
        i+=1
    
    #plt.show()
    exp_dir = './experiments/rda_decision_tree'
    plt.savefig(exp_dir+"/rda_dataframe"+sheet+".png", dpi=300)
    plt.close()


def draw_dt_result_compare_1(rows, y_true, y_pred):
    plt.rcParams['figure.figsize'] = [7, 5]
    x = list(range(int(rows)))
    plt.plot(y_true, label='Measured(GT)', color='y')
    plt.scatter(x, y_pred, label='Predicted', marker='o', color='b')
    plt.legend()
    #plt.show()
    exp_dir = './experiments/rda_decision_tree'
    plt.savefig(exp_dir+"/groundtrue_vs_predictions_1.png", dpi=300)
    plt.close()


def draw_dt_result_compare_2(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, marker='o', color='b', s=70)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=4)
    ax.set_xlabel('Measured(GT)')
    ax.set_ylabel('Predicted')
    #plt.show()
    exp_dir = './experiments/rda_decision_tree'
    plt.savefig(exp_dir+"/groundtrue_vs_predictions_2.png", dpi=300)
    plt.close(fig)


def draw_dt_feature_importance(model):
    # plot the feature importance
    xgb.plot_importance(model)
    plt.rcParams['figure.figsize'] = (100, 50) # widht, height
    #plt.show()
    exp_dir = './experiments/rda_decision_tree'
    plt.savefig(exp_dir+"/feature_importance.png", dpi=300)
    plt.close()


def draw_bl_losses(train_loss, val_loss):
    plt.rcParams['figure.figsize'] = [7, 5]
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='cross validation')
    plt.legend()
    #plt.show()
    exp_dir = './experiments/rda_basic_lstm'
    plt.savefig(exp_dir+"/loss_epoch", dpi=300)
    plt.close()


def draw_bl_result_compare_1(rows, y_true, y_pred):
    plt.rcParams['figure.figsize'] = [7, 5]
    x = list(range(int(rows)))
    plt.plot(y_true, label='Measured(GT)', color='y')
    plt.scatter(x, y_pred, label='Predicted', marker='o', color='b')
    plt.legend()
    #plt.show()
    exp_dir = './experiments/rda_basic_lstm'
    plt.savefig(exp_dir+"/groundtrue_vs_predictions_1.png", dpi=300)
    plt.close()


def draw_bl_result_compare_2(inv_y_true, inv_y_pred):
    fig, ax = plt.subplots()
    ax.scatter(inv_y_true, inv_y_pred, marker='o', color='b', s=100)
    ax.plot([inv_y_true.min(), inv_y_true.max()], [inv_y_true.min(), inv_y_true.max()], lw=4)
    ax.set_xlabel('Measured(GT)')
    ax.set_ylabel('Predicted')
    #plt.show()
    exp_dir = './experiments/rda_basic_lstm'
    plt.savefig(exp_dir+"/groundtrue_vs_predictions_2.png", dpi=300)
    plt.close(fig)
