import os
import json
from collections import defaultdict
from flask import Flask, render_template, request


app = Flask(__name__)


# You can override the metadata folder by passing an env
META_DATA_FOLDER = os.environ.get('META_DATA_FOLDER', '../.reclist')


def read_data_from_report(reclist_name: str, model_name: str, run_time: str, verbose: bool=False):
    target_path = os.path.join(META_DATA_FOLDER, reclist_name, model_name, run_time)
    with open(os.path.join(target_path, 'results', 'report.json')) as f:
        data = json.load(f)

    if verbose:
        print("\nResults loaded: {}\n".format(data))

    return data


def get_list_of_runs(data_folder: str):
    rec_table = []
    # get reclist
    all_reclists = [f for f in os.scandir(data_folder) if f.is_dir()]    
    for r in all_reclists:
        # now get the models
        all_models = [f for f in os.scandir(r.path) if f.is_dir()] 
        for m in all_models:
            # now get the runs
            all_runs = [f.name for f in os.scandir(m.path) if f.is_dir()] 
            for idx, run in enumerate(all_runs):
                rec_table.append({
                        "reclist_name": r.name,
                        "model_name": m.name,
                        "run_int": idx + 1, # avoid a 0-based run
                        "run_time": run,
                        "path": '{}/{}/{}'.format(r.name, m.name, run)
                    })
 
    return rec_table


def get_artifacts(reclist_to_report: dict):
    # TODO: when we store artifacts, revisit this, now static
    artifacts = [
        {
            'name': list(reclist_to_report.values())[0]['model_name'],
            'type': 'Model',
            'path': '/Users/jacopo/repos/RecList/public/examples/recList_tests'
        },
        {
            'name': 'Coveo Dataset',
            'type': 'Dataset',
            'path': '/Users/jacopo/repos/RecList/public/examples/recList_tests'
        }
    ]

    return artifacts


@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        # debug
        # print("Method is GET")
        # if get, return list of available RecList objects
        rec_table = get_list_of_runs(META_DATA_FOLDER)
        return render_template('index.html', reclist_table=rec_table)   

    if request.method == 'POST':
        # if POST, run the analysis with one or more tests to display
        # debug
        # print("Method is POST: data received is {}".format(request.form))
        reclists_to_analyze = [val for key, val in request.form.items()]
        if not reclists_to_analyze:
            # TODO: handle this ;-)
            raise Exception("No list selected!!!")
        # debug 
        print("Building a page for lists: {}".format(','.join(reclists_to_analyze)))
        reclist_to_report = {
            _r: {
                    "path": _r,
                    "reclist_name": _r.split('/')[0],
                    "model_name": _r.split('/')[1],
                    "run_time": _r.split('/')[2],
                    "report": read_data_from_report(
                        reclist_name=_r.split('/')[0],
                        model_name=_r.split('/')[1],
                        run_time=_r.split('/')[2]
                    )
                } 
            for _r in reclists_to_analyze
        }
        # check if all models have = reclist, otherwise is pointless to compare
        reclist_names = set(_r.split('/')[0] for _r in reclists_to_analyze)
        if len(reclist_names) > 1:
            # TODO: handle this ;-)
            raise Exception("All tests need to have the same RecList for a fair comparison!")
        reclist_name = list(reclist_names)[0]
        # populate a data object to fill the template
        models = get_models_from_report(reclist_to_report)
        artifacts = get_artifacts(reclist_to_report)
        table = get_table_from_report(models, reclist_to_report)
        data = {
            'reclist_name': reclist_name,
            'models': models,
            'artifacts': artifacts,
            'result_table': table,
            'charts': get_charts_from_report(models, reclist_to_report)
        }

        return render_template('reclist.html', data=data)


def get_charts_from_report(models: list, report: dict):
    #  we use the last three digit of epoch time and model name as an id
    model_ids = ['{} ({})'.format(m['model_name'], m['run_time'][-3:]) for m in models]
    _labels = []
    _datasets = []
    # loop over reports
    for _, _report in report.items():
        # for each report loop over test results 
        model_id = '{} ({})'.format(_report['model_name'], _report['run_time'][-3:])
        for _r in _report['report']['data']:
            if _r["test_name"] == "brand_cosine_distance":
                _counts = _r["test_result"]["histogram"][0]
                _bins = _r["test_result"]["histogram"][1]
                # TODO: guarantee bins are the same, maybe we should do binning front-end instead
                _labels = _bins
                _datasets.append(
                    {
                        'label': model_id,
                        'data': _counts
                    }
                )

    return {
        'brand_cosine_distance': {
            'labels': _labels,
            'datasets': _datasets
        }
    }


def get_table_from_report(models: list, report: dict):
    model_to_results  = defaultdict(dict)
    all_tests =  {}
    #  we use the last three digit of epoch time and model name as an id
    model_ids = ['{} ({})'.format(m['model_name'], m['run_time'][-3:]) for m in models]
    cols = ['Name', 'Description'] +  model_ids
    # loop over reports
    for _, _report in report.items():
        # for each report loop over test results 
        model_id = '{} ({})'.format(_report['model_name'], _report['run_time'][-3:])
        for _r in _report['report']['data']:
            # if it's a chart, ignore it here
            if type(_r["test_result"]) is not float: 
                continue
            # keep track of all tests
            if _r['test_name'] not in all_tests:
                all_tests[_r['test_name']] = _r['description']
            # record the model score on this test
            model_to_results[model_id][_r['test_name']] = _r["test_result"]
    # prepare the final rows
    rows = []
    for _name, _description in all_tests.items():
        cnt_row = [ _name, _description ]
        for m in model_ids:
            cnt_row.append(model_to_results[m][_name])
        # append the rows
        rows.append(cnt_row)

    # debug
    print(rows)

    return {
        'cols': cols,
        'rows': rows
    }


def get_models_from_report(report: dict):
    return [
                {
                    'path': r['path'],
                    'model_name': r['model_name'],
                    'run_time': r['run_time'],
                    'description': 'Test description for now'
                } for _, r  in report.items()
            ]



if __name__ == "__main__":
    app.debug = True
    app.run()