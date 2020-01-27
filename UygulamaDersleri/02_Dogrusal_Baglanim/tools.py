# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:55:19 2020
@author: utkuk
fonksiyonları şu linkten aldım: https://emredjan.github.io/blog/2017/07/11/emulating-r-plots-in-python/
"""
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
import numpy as np

def requiredValues(fitted):
    # fitted values (need a constant term for intercept)
    model_fitted_y = fitted.fittedvalues
    
    # model residuals
    model_residuals = fitted.resid
    
    # normalized residuals
    model_norm_residuals = fitted.get_influence().resid_studentized_internal
    
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
    
    # leverage, from statsmodels internals
    model_leverage = fitted.get_influence().hat_matrix_diag
    
    # cook's distance, from statsmodels internals
    model_cooks = fitted.get_influence().cooks_distance[0]
    
    return model_fitted_y, model_residuals, model_norm_residuals, model_norm_residuals_abs_sqrt, model_abs_resid, model_leverage, model_cooks


def residualPlot(fitted, target, df):
    # residual plot
    model_abs_resid = np.abs(fitted.resid)
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]
    
    plot_lm_1 = plt.figure(1)
    plot_lm_1.set_figheight(8)
    plot_lm_1.set_figwidth(12)
    
    plot_lm_1.axes[0] = sns.residplot(fitted.fittedvalues, target, data=df,
                                      lowess=True,
                                      scatter_kws={'alpha': 0.5},
                                      line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm_1.axes[0].set_title('Residuals vs Fitted')
    plot_lm_1.axes[0].set_xlabel('Fitted values')
    plot_lm_1.axes[0].set_ylabel('Residuals') 
    for i in abs_resid_top_3.index:
        plot_lm_1.axes[0].annotate(i, 
                                   xy=(fitted.fittedvalues[i], 
                                       fitted.resid[i]))
    return plot_lm_1


def qqPlot(fitted):
    QQ = ProbPlot(fitted.get_influence().resid_studentized_internal)
    plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
    
    plot_lm_2.set_figheight(8)
    plot_lm_2.set_figwidth(12)
    
    plot_lm_2.axes[0].set_title('Normal Q-Q')
    plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
    plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
    
    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(fitted.get_influence().resid_studentized_internal)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    
    for r, i in enumerate(abs_norm_resid_top_3):
        plot_lm_2.axes[0].annotate(i, 
                                   xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                       fitted.get_influence().resid_studentized_internal[i]))
    return plot_lm_2


def scaleLocationPlot(fitted, model_fitted_y, model_norm_residuals_abs_sqrt):
    plot_lm_3 = plt.figure(3)
    plot_lm_3.set_figheight(8)
    plot_lm_3.set_figwidth(12)
    
    plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    
    plot_lm_3.axes[0].set_title('Scale-Location')
    plot_lm_3.axes[0].set_xlabel('Fitted values')
    plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');
    
    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(fitted.get_influence().resid_studentized_internal)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    for i in abs_norm_resid_top_3:
        plot_lm_3.axes[0].annotate(i, 
                                   xy=(model_fitted_y[i], 
                                       model_norm_residuals_abs_sqrt[i]))
    return plot_lm_3


def leveragePlot(fitted, model_leverage, model_norm_residuals, model_cooks):
    plot_lm_4 = plt.figure(4)
    plot_lm_4.set_figheight(8)
    plot_lm_4.set_figwidth(12)
    
    plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(model_leverage, model_norm_residuals, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    
    plot_lm_4.axes[0].set_xlim(0, 0.20)
    plot_lm_4.axes[0].set_ylim(-3, 5)
    plot_lm_4.axes[0].set_title('Residuals vs Leverage')
    plot_lm_4.axes[0].set_xlabel('Leverage')
    plot_lm_4.axes[0].set_ylabel('Standardized Residuals')
    
    # annotations
    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
    
    for i in leverage_top_3:
        plot_lm_4.axes[0].annotate(i, 
                                   xy=(model_leverage[i], 
                                       model_norm_residuals[i]))
        
    # shenanigans for cook's distance contours
    def graph(formula, x_range, label=None):
        x = x_range
        y = formula(x)
        plt.plot(x, y, label=label, lw=1, ls='--', color='red')
    
    p = len(fitted.params) # number of model parameters
    
    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
          np.linspace(0.001, 0.200, 50), 
          'Cook\'s distance') # 0.5 line
    
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
          np.linspace(0.001, 0.200, 50)) # 1 line
    
    plt.legend(loc='upper right')
    return plot_lm_4


def plots(fitted, target, df):
    # required plots
    model_fitted_y, model_residuals, model_norm_residuals, model_norm_residuals_abs_sqrt, model_abs_resid, model_leverage, model_cooks = requiredValues(fitted)
    
    # residualPlot
    residualPlot(fitted, target, df)
    
    # qq plot
    qqPlot(fitted)
    
    # scale-location plot
    scaleLocationPlot(fitted, model_fitted_y, model_norm_residuals_abs_sqrt)
        
    # Leverage plot
    leveragePlot(fitted, model_leverage, model_norm_residuals, model_cooks)
