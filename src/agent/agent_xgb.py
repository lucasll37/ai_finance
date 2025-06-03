import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import XgBoost
from IPython.display import display, clear_output

class Agent_xgb:
    
    def __init__(self, data, drop_intersection_time_series= 5, despise=[], **kwargs):
        self.model = XgBoost(target="Trend")
        self.despise = despise
        self.features_name = data.iloc[:, len(self.despise):]

        self.model.build(
            data,
            # task="classification",
            task="regression",
            max_cat_nunique = 10,
            split_size = (0.7, 0.15, 0.15),
            patience_early_stopping = 10,
            shuffle_split = False,
            drop_intersection_time_series = drop_intersection_time_series,
            despise=self.despise,
            **kwargs
        )

    def load(self, foldername, path="../../saved"):
        self.model.load(foldername, path)

    def optimizer(self, n_trials=10):
        self.model.hyperparameter_optimization(
            n_trials=n_trials,
            num_folds = 5, 
            info=True,
            shuffle_kfold = False,
            search_space_tree_method = ['auto'],
            search_space_booster = ['gbtree', 'gblinear', 'dart'],
            search_space_learning_rate = [0.1, 0.15, 0.2, 0.25, 0.3],
            search_space_min_split_loss = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            search_space_max_depth = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            search_space_min_child_weight = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            search_space_max_delta_step = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            search_space_subsample = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            search_space_sampling_method = ['uniform'],
            search_space_colsample_bytree = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            search_space_colsample_bylevel = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            search_space_colsample_bynode = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            search_space_reg_lambda = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            search_space_reg_alpha = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            search_space_scale_pos_weight = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            search_space_grow_policy = ['depthwise', 'lossguide'],
            search_space_max_leaves = [0, 1, 2, 3, 4],
            search_space_max_bin = [32, 64, 128, 256, 512, 1024],
            search_space_num_parallel_tree = [1, 2, 3, 4]
        )

    def fit(self, path="./saved"):
        self.model.fit(return_history=True, graphic=True, graphic_save_extension='png', path=path, verbose=0, shuffle_kfold=False)
        self.model.save(path=path)

    def predict(self, data, return_df=False, graphic=True, graphic_save_extension='png', path="./saved/figures", delay=1):  
        
        prediction = self.model.predict(data)

        state = 0
        wait = 0
        
        open_price = None
        close_price = None
        start_date = None
        close_date = None

        result = pd.DataFrame(columns=["Start Date", "Close Date", "Duration", "Open Price", "Close Price", "Return", "Profit"])

        for index, row in prediction[['_Close', 'Trend (XGB prediction)']].iterrows():

            ################ IN ACTION ####################
            if state == 1:
                if row['Trend (XGB prediction)'] <= 0.5:
                    wait += 1

                else:
                    wait = 0

                if wait == delay:
                    wait = 0
                    state = 0
                    close_price = row['_Close']
                    close_date = index
                    duration = (close_date - start_date).days
                    profit = close_price - open_price
                    operation = pd.DataFrame({
                        "Start Date": [start_date],
                        "Close Date":[close_date],
                        "Duration": [duration],
                        "Open Price": [open_price],
                        "Close Price":[close_price],
                        "Return": [profit/open_price],
                        "Profit": [profit],
                    })

                    result = pd.concat([result ,operation])

            else:
                ################ NO ACTION ####################
                if row['Trend (XGB prediction)'] > 0.5:
                    wait += 1

                else:
                    wait = 0

                if wait == delay:
                    state = 1
                    wait = 0
                    open_price = row['_Close']
                    start_date = index

        if state == 1:
            close_price = row['_Close']
            close_date = index
            duration = (close_date - start_date).days
            profit = close_price - open_price
            operation = pd.DataFrame({
                "Start Date": [start_date],
                "Close Date":[close_date],
                "Duration": [duration],
                "Open Price": [open_price],
                "Close Price":[close_price],
                "Return": [profit/open_price],
                "Profit": [profit],
            })

            result = pd.concat([result, operation])

        result['Return Acc (%)'] = result['Return'].cumsum()        
        result.reset_index(drop=True, inplace=True)

            ########### GRAPHIC ################
        if graphic:
            fig, axs = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]}, dpi=100)

            sns.lineplot(data, x="Date", y="_Close", label="Close", linestyle='-', linewidth=1, color="blue", alpha=1, ax=axs[0], legend=False)
            axs[0].set_xlabel('Date')
            axs[0].set_ylabel('Price')

            ax2 = axs[1].twinx()
            axs = np.append(axs, ax2)

            sns.lineplot(data, x="Date", y="Trend", label="Trend", linestyle='-', linewidth=3, color="black", alpha=1, ax=axs[1], legend=False)
            sns.lineplot(prediction, x="Date", y="Trend (XGB prediction)", label="Predict", linestyle='-', linewidth=1, color="red", alpha=1, ax=axs[2], legend=False)

            axs[1].set_ylabel('Uptrand')
            axs[1].set_ylabel('Trend')
            axs[2].set_ylabel('Trend (XGB prediction)', color='r')

            for _, row in result.iterrows():
                color = 'green' if row['Profit'] > 0 else 'red'
                sns.lineplot(x=[row['Start Date'], row['Close Date']], y=[row['Open Price'], row['Close Price']], linestyle='--', color=color, ax=axs[0])


            sns.scatterplot(data=result, x="Start Date", y="Open Price", color='green', label='Entrada', marker='^', s=100, ax=axs[0])
            sns.scatterplot(data=result, x="Close Date", y="Close Price", color='red', label='Saída', marker='v', s=100, ax=axs[0])

            _y = [0, 1]
            _labels = {0: 'Down/Neutral', 1: 'Up'}
            axs[1].set_yticks(_y)
            axs[1].set_yticklabels([_labels[i] for i in _y])
            axs[2].set_yticks(_y)
            axs[2].set_yticklabels([_labels[i] for i in _y])
            axs[2].tick_params(axis='y', labelcolor='r')

            lines1, labels1 = axs[0].get_legend_handles_labels()
            lines2, labels2 = axs[1].get_legend_handles_labels()
            lines3, labels3 = axs[2].get_legend_handles_labels()
            axs[0].legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')


            for index, row in result.iterrows():
                color = 'green' if row['Profit'] > 0 else 'red'
                axs[0].fill_between(data.loc[row["Start Date"]:row["Close Date"]].index, data.loc[row["Start Date"]:row["Close Date"], "_Close"].max(), data.loc[row["Start Date"]:row["Close Date"], "_Close"].min(), color=color, alpha=0.2)

            
            _return = 0

            if not result.empty:
                years = (result["Close Date"].iloc[-1] - result["Start Date"].iloc[0]).days / 365.25
                _return  = (((1 + result['Return'].sum()) ** (1/years)) - 1) * 100
            
            fig.suptitle(f'Results ({_return:.2f} % a.a.)', fontsize=16)

            plt.tight_layout(rect=[0, 0, 1, 0.98])

            plt.savefig(f'{path}/figures/Results_Graphic.{graphic_save_extension}', format=f'{graphic_save_extension}')
            plt.show()
            plt.close()
            ########### GRAPHIC ################

        if return_df:
            return result
        
    def naive_choice(self, data, signal, policy, return_df=False, graphic=True, graphic_save_extension='png', path="./saved/figures", delay=1):  

        state = 0
        wait = 0
        
        open_price = None
        close_price = None
        start_date = None
        close_date = None

        result = pd.DataFrame(columns=["Start Date", "Close Date", "Duration", "Open Price", "Close Price", "Return", "Profit"])

        for index, row in data[['_Close', signal]].iterrows():

            ################ IN ACTION ####################
            if state == 1:
                if policy['out'](row[signal]):
                    wait += 1

                else:
                    wait = 0

                if wait == delay:
                    wait = 0
                    state = 0
                    close_price = row['_Close']
                    close_date = index
                    duration = (close_date - start_date).days
                    profit = close_price - open_price
                    operation = pd.DataFrame({
                        "Start Date": [start_date],
                        "Close Date":[close_date],
                        "Duration": [duration],
                        "Open Price": [open_price],
                        "Close Price":[close_price],
                        "Return": [profit/open_price],
                        "Profit": [profit],
                    })

                    result = pd.concat([result, operation])

            else:
                ################ NO ACTION ####################
                if policy['entry'](row[signal]):
                    wait += 1

                else:
                    wait = 0

                if wait == delay:
                    state = 1
                    wait = 0
                    open_price = row['_Close']
                    start_date = index

        if state == 1:
            close_price = row['_Close']
            close_date = index
            duration = (close_date - start_date).days
            profit = close_price - open_price
            operation = pd.DataFrame({
                "Start Date": [start_date],
                "Close Date":[close_date],
                "Duration": [duration],
                "Open Price": [open_price],
                "Close Price":[close_price],
                "Return": [profit/open_price],
                "Profit": [profit],
            })
            
            result = pd.concat([result, operation])

        result['Return Acc (%)'] = result['Return'].cumsum()    
        result.reset_index(drop=True, inplace=True)
        
        _return = 0

        if not result.empty:
            years = (result["Close Date"].iloc[-1] - result["Start Date"].iloc[0]).days / 365.25
            _return  = (((1 + result['Return'].sum()) ** (1/years)) - 1) * 100

            ########### GRAPHIC ################
        if graphic:
            fig, axs = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]}, dpi=100)

            sns.lineplot(data, x="Date", y="_Close", label="Close", linestyle='-', linewidth=1, color="blue", alpha=1, ax=axs[0], legend=False)
            axs[0].set_xlabel('Date')
            axs[0].set_ylabel('Price')

            ax2 = axs[1].twinx()
            axs = np.append(axs, ax2)

            sns.lineplot(data, x="Date", y="Trend", label="Trend", linestyle='-', linewidth=3, color="black", alpha=1, ax=axs[1], legend=False)
            sns.lineplot(data, x="Date", y=signal, label=signal, linestyle='-', linewidth=1, color="red", alpha=1, ax=axs[2], legend=False)

            axs[1].set_ylabel('Uptrand')
            axs[1].set_ylabel('Trend')
            axs[2].set_ylabel(signal, color='r')

            for i, row in result.iterrows():
                color = 'green' if row['Profit'] > 0 else 'red'
                sns.lineplot(x=[row['Start Date'], row['Close Date']], y=[row['Open Price'], row['Close Price']], linestyle='--', color=color, ax=axs[0])
                

            sns.scatterplot(data=result, x="Start Date", y="Open Price", color='green', label='Entrada', marker='^', s=100, ax=axs[0])
            sns.scatterplot(data=result, x="Close Date", y="Close Price", color='red', label='Saída', marker='v', s=100, ax=axs[0])

            _y = [0, 1]
            _labels = {0: 'Down/Neutral', 1: 'Up'}
            axs[1].set_yticks(_y)
            axs[1].set_yticklabels([_labels[i] for i in _y])
            axs[2].tick_params(axis='y', labelcolor='r')

            lines1, labels1 = axs[0].get_legend_handles_labels()
            lines2, labels2 = axs[1].get_legend_handles_labels()
            lines3, labels3 = axs[2].get_legend_handles_labels()
            axs[0].legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')


            for index, row in result.iterrows():
                color = 'green' if row['Profit'] > 0 else 'red'
                # axs[0].fill_between(data.loc[row["Start Date"]:row["Close Date"]].index, data.loc[row["Start Date"]:row["Close Date"], "_Close"], 0, color=color, alpha=0.2)
                axs[0].fill_between(data.loc[row["Start Date"]:row["Close Date"]].index, data.loc[row["Start Date"]:row["Close Date"], "_Close"].max(), data.loc[row["Start Date"]:row["Close Date"], "_Close"].min(), color=color, alpha=0.2)

            # _return = 0

            # if not result.empty:
            #     years = (result["Close Date"].iloc[-1] - result["Start Date"].iloc[0]).days / 365.25
            #     _return  = (((1 + result['Return'].sum()) ** (1/years)) - 1) * 100
                
            

            fig.suptitle(f'Results with naive choice {signal} ({_return:.2f} % a.a.)', fontsize=16)

            plt.tight_layout(rect=[0, 0, 1, 0.98])

            # plt.savefig(f'{path}/figures/Results.{graphic_save_extension}', format=f'{graphic_save_extension}')
            plt.show()
            plt.close()
            ########### GRAPHIC ################

        if return_df:
            return result
        
        else:
            return _return
        

    def naive_choice_market(self, companies, return_df=False): 

        all_market = pd.DataFrame(columns=["Ativo", "Signal", "Fisrt Open Oper.", "Last Close Oper.", "Return (% a.a.)"])

        for company in companies:

            try:
                ticket = company['ticket']
                data = company['data']

                for signal, values in company['signals'].items():
                    policy = values['policy']
                    delay = values['delay']

                    state = 0
                    wait = 0
                    
                    open_price = None
                    close_price = None
                    start_date = None
                    close_date = None

                    result = pd.DataFrame(columns=["Start Date", "Close Date", "Duration", "Open Price", "Close Price", "Return", "Profit"])

                    for index, row in data[['_Close', signal]].iterrows(): 

                        ################ IN ACTION ####################
                        if state == 1:
                            if policy['out'](row[signal]):
                                wait += 1

                            else:
                                wait = 0

                            if wait == delay:
                                wait = 0
                                state = 0
                                close_price = row['_Close']
                                close_date = index
                                duration = (close_date - start_date).days
                                profit = close_price - open_price
                                operation = pd.DataFrame({
                                    "Start Date": [start_date],
                                    "Close Date":[close_date],
                                    "Duration": [duration],
                                    "Open Price": [open_price],
                                    "Close Price":[close_price],
                                    "Return": [profit/open_price],
                                    "Profit": [profit],
                                })

                                result = pd.concat([result, operation])

                        else:
                            ################ NO ACTION ####################
                            if policy['entry'](row[signal]):
                                wait += 1

                            else:
                                wait = 0

                            if wait == delay:
                                state = 1
                                wait = 0
                                open_price = row['_Close']
                                start_date = index

                    if state == 1:
                        close_price = row['_Close']
                        close_date = index
                        duration = (close_date - start_date).days
                        profit = close_price - open_price
                        operation = pd.DataFrame({
                            "Start Date": [start_date],
                            "Close Date":[close_date],
                            "Duration": [duration],
                            "Open Price": [open_price],
                            "Close Price":[close_price],
                            "Return": [profit/open_price],
                            "Profit": [profit],
                        })
                    
                        result = pd.concat([result, operation])

                    result['Return Acc (%)'] = result['Return'].cumsum()            
                    result.reset_index(drop=True, inplace=True)


                    if not result.empty:
                        years = (result["Close Date"].iloc[-1] - result["Start Date"].iloc[0]).days / 365.25
                        _return  = (((1 + result['Return'].sum()) ** (1/years)) - 1) * 100
                        new_value = [ticket, signal, result["Start Date"].iloc[0], result["Close Date"].iloc[-1], _return]

                    else:
                        _return = 0
                        new_value = [ticket, signal, None, None, _return]

                    new_register = pd.DataFrame([new_value], columns=all_market.columns)
                    all_market = pd.concat([all_market, new_register], ignore_index=True)

                    clear_output(wait=True)
                    display(all_market)

            except Exception as e:
                clear_output(wait=True)
                print(f"Problem with ticket: {ticket}: {signal}: {e}")
                return


        if return_df:
            return all_market
    

    def combined_choice_market(self, companies, return_df=False):

        all_market = pd.DataFrame(columns=["Ativo", "Signal", "Fisrt Open Oper.", "Last Close Oper.", "Return (% a.a.)"])
        
        for company in companies:
            ticket = company['ticket']
            data = company['data']
            delay = company['delay']
            min_entry = company['min_entry']
            min_out = company['min_out']

            state = 0
            wait = 0
            
            open_price = None
            close_price = None
            start_date = None
            close_date = None
            combined_signals = " + ".join(list(company['signals'].keys()))

            result = pd.DataFrame(columns=["Start Date", "Close Date", "Duration", "Open Price", "Close Price", "Return", "Profit"])

            for index, row in data.iterrows(): 

                entry = 0
                out = 0

                for signal, values in company['signals'].items():
                    policy = values['policy']

                    if policy['entry'](row[signal]):
                        entry += 1

                    if policy['out'](row[signal]):
                        out += 1

                ################ IN ACTION ####################
                if state == 1:
                    if out >= min_out:
                        wait += 1

                    else:
                        wait = 0

                    if wait == delay:
                        wait = 0
                        state = 0
                        close_price = row['_Close']
                        close_date = index
                        duration = (close_date - start_date).days
                        profit = close_price - open_price
                        operation = pd.DataFrame({
                            "Start Date": [start_date],
                            "Close Date":[close_date],
                            "Duration": [duration],
                            "Open Price": [open_price],
                            "Close Price":[close_price],
                            "Return": [profit/open_price],
                            "Profit": [profit],
                        })

                        result = pd.concat([result, operation])

                else:
                    ################ NO ACTION ####################
                    if entry >= min_entry:
                        wait += 1

                    else:
                        wait = 0

                    if wait == delay:
                        state = 1
                        wait = 0
                        open_price = row['_Close']
                        start_date = index

            if state == 1:
                close_price = row['_Close']
                close_date = index
                duration = (close_date - start_date).days
                profit = close_price - open_price
                operation = pd.DataFrame({
                    "Start Date": [start_date],
                    "Close Date":[close_date],
                    "Duration": [duration],
                    "Open Price": [open_price],
                    "Close Price":[close_price],
                    "Return": [profit/open_price],
                    "Profit": [profit],
                })

                result = pd.concat([result, operation])

            result['Return Acc (%)'] = result['Return'].cumsum()
            result.reset_index(drop=True, inplace=True)


            if not result.empty:
                years = (result["Close Date"].iloc[-1] - result["Start Date"].iloc[0]).days / 365.25
                _return  = (((1 + result['Return'].sum()) ** (1/years)) - 1) * 100
                new_value = [ticket, combined_signals, result["Start Date"].iloc[0], result["Close Date"].iloc[-1], _return]

            else:
                _return = 0
                new_value = [ticket, combined_signals, None, None, _return]


            new_register = pd.DataFrame([new_value], columns=all_market.columns)
            all_market = pd.concat([all_market, new_register], ignore_index=True)

            clear_output(wait=True)
            display(all_market)

        if return_df:
            return all_market
        
        
    def ai_choice_market(self, companies, return_df=False):

        all_market = pd.DataFrame(columns=["Ativo", "Signal", "Fisrt Open Oper.", "Last Close Oper.", "Return (% a.a.)"])
        
        for company in companies:
            ticket = company['ticket']
            data = company['data']
            delay = company['delay']

            prediction = self.model.predict(data)

            state = 0
            wait = 0
            
            open_price = None
            close_price = None
            start_date = None
            close_date = None

            result = pd.DataFrame(columns=["Start Date", "Close Date", "Duration", "Open Price", "Close Price", "Return", "Profit"])

            for index, row in prediction[['_Close', 'Trend (XGB prediction)']].iterrows():

                ################ IN ACTION ####################
                if state == 1:
                    if row['Trend (XGB prediction)'] <= 0.5:
                        wait += 1

                    else:
                        wait = 0

                    if wait == delay:
                        wait = 0
                        state = 0
                        close_price = row['_Close']
                        close_date = index
                        duration = (close_date - start_date).days
                        profit = close_price - open_price
                        operation = pd.DataFrame({
                            "Start Date": [start_date],
                            "Close Date":[close_date],
                            "Duration": [duration],
                            "Open Price": [open_price],
                            "Close Price":[close_price],
                            "Return": [profit/open_price],
                            "Profit": [profit],
                        })

                        result = pd.concat([result, operation])

                else:
                    ################ NO ACTION ####################
                    if row['Trend (XGB prediction)'] > 0.5:
                        wait += 1

                    else:
                        wait = 0

                    if wait == delay:
                        state = 1
                        wait = 0
                        open_price = row['_Close']
                        start_date = index

            if state == 1:
                close_price = row['_Close']
                close_date = index
                duration = (close_date - start_date).days
                profit = close_price - open_price
                operation = pd.DataFrame({
                    "Start Date": [start_date],
                    "Close Date":[close_date],
                    "Duration": [duration],
                    "Open Price": [open_price],
                    "Close Price":[close_price],
                    "Return": [profit/open_price],
                    "Profit": [profit],
                })

                result = pd.concat([result, operation])

            result['Return Acc (%)'] = result['Return'].cumsum()
            result.reset_index(drop=True, inplace=True)


            if not result.empty:
                years = (result["Close Date"].iloc[-1] - result["Start Date"].iloc[0]).days / 365.25
                _return  = (((1 + result['Return'].sum()) ** (1/years)) - 1) * 100
                new_value = [ticket, "XG Boost", result["Start Date"].iloc[0], result["Close Date"].iloc[-1], _return]

            else:
                _return = 0
                new_value = [ticket, "XG Boost", None, None, _return]


            new_register = pd.DataFrame([new_value], columns=all_market.columns)
            all_market = pd.concat([all_market, new_register], ignore_index=True)

            clear_output(wait=True)
            display(all_market)

        if return_df:
            return all_market
        

    def optimum_choice(self, data, threshold=0.5, return_df=False, graphic=True, delay=1):  
            
            state = 0
            wait = 0
            
            open_price = None
            close_price = None
            start_date = None
            close_date = None

            result = pd.DataFrame(columns=["Start Date", "Close Date", "Duration", "Open Price", "Close Price", "Return", "Profit"])

            for index, row in data[['_Close', 'Trend']].iterrows():

                ################ IN ACTION ####################
                if state == 1:
                    if row["Trend"] <= threshold:
                        wait += 1

                    else:
                        wait = 0

                    if wait == delay:
                        wait = 0
                        state = 0
                        close_price = row['_Close']
                        close_date = index
                        duration = (close_date - start_date).days
                        profit = close_price - open_price
                        operation = pd.DataFrame({
                            "Start Date": [start_date],
                            "Close Date":[close_date],
                            "Duration": [duration],
                            "Open Price": [open_price],
                            "Close Price":[close_price],
                            "Return": [profit/open_price],
                            "Profit": [profit],
                        })
                        
                        result = pd.concat([result, operation])

                else:
                    ################ NO ACTION ####################
                    if row["Trend"] > threshold:
                        wait += 1

                    else:
                        wait = 0

                    if wait == delay:
                        state = 1
                        wait = 0
                        open_price = row['_Close']
                        start_date = index

            if state == 1:
                close_price = row['_Close']
                close_date = index
                duration = (close_date - start_date).days
                profit = close_price - open_price
                operation = pd.DataFrame({
                    "Start Date": [start_date],
                    "Close Date":[close_date],
                    "Duration": [duration],
                    "Open Price": [open_price],
                    "Close Price":[close_price],
                    "Return": [profit/open_price],
                    "Profit": [profit],
                })

                result = pd.concat([result, operation])

            result['Return Acc (%)'] = result['Return'].cumsum()            
            result.reset_index(drop=True, inplace=True)

                ########### GRAPHIC ################
            if graphic:
                fig, axs = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]}, dpi=100)

                sns.lineplot(data, x="Date", y="_Close", label="Close", linestyle='-', linewidth=1, color="blue", alpha=1, ax=axs[0], legend=False)
                axs[0].set_xlabel('Date')
                axs[0].set_ylabel('Price')

                sns.lineplot(data, x="Date", y="Trend", label="Trend", linestyle='-', linewidth=3, color="black", alpha=1, ax=axs[1], legend=False)

                axs[1].set_ylabel('Uptrand')
                axs[1].set_ylabel('Trend')

                for i, row in result.iterrows():
                    color = 'green' if row['Profit'] > 0 else 'red'
                    sns.lineplot(x=[row['Start Date'], row['Close Date']], y=[row['Open Price'], row['Close Price']], linestyle='--', color=color, ax=axs[0])

                sns.scatterplot(data=result, x="Start Date", y="Open Price", color='green', label='Entrada', marker='^', s=100, ax=axs[0])
                sns.scatterplot(data=result, x="Close Date", y="Close Price", color='red', label='Saída', marker='v', s=100, ax=axs[0])

                _y = [0, 1]
                _labels = {0: 'Down/Neutral', 1: 'Up'}
                axs[1].set_yticks(_y)
                axs[1].set_yticklabels([_labels[i] for i in _y])

                lines1, labels1 = axs[0].get_legend_handles_labels()
                lines2, labels2 = axs[1].get_legend_handles_labels()
                axs[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')


                for index, row in result.iterrows():
                    color = 'green' if row['Profit'] > 0 else 'red'
                    # axs[0].fill_between(data.loc[row["Start Date"]:row["Close Date"]].index, data.loc[row["Start Date"]:row["Close Date"], "_Close"], 0, color=color, alpha=0.2)
                    axs[0].fill_between(data.loc[row["Start Date"]:row["Close Date"]].index, data.loc[row["Start Date"]:row["Close Date"], "_Close"].max(), data.loc[row["Start Date"]:row["Close Date"], "_Close"].min(), color=color, alpha=0.2)

                    
                _return = 0

                if not result.empty:
                    years = (result["Close Date"].iloc[-1] - result["Start Date"].iloc[0]).days / 365.25
                    _return  = (((1 + result['Return'].sum()) ** (1/years)) - 1) * 100

                fig.suptitle(f'Results optimum ({_return:.2f} % a.a.)', fontsize=16)

                plt.tight_layout(rect=[0, 0, 1, 0.98])
                plt.show()
                plt.close()
                ########### GRAPHIC ################

            if return_df:
                return result


    def feature_importances(self):
        feature_importances = self.model.model.feature_importances_
        sorted_idx = np.argsort(feature_importances)[::-1]
        sorted_feature_names = self.features_name.columns[sorted_idx]

        plt.figure(figsize=(10, 5), dpi=100)
        plt.barh(sorted_feature_names, feature_importances[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Name')
        plt.title('Feature Importances')
        plt.gca().invert_yaxis()
        plt.show()
        plt.close()
