#!/usr/bin/env python3

import argparse
import os
import subprocess
from glob import glob
import pandas as pd

__version__ = "1.0.0"

def run(command, env={}):
    merged_env = os.environ
    merged_env.update(env)
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True,
                               env=merged_env)
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() != None:
            break
    if process.returncode != 0:
        raise Exception("Non zero return code: %d"%process.returncode)

parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files '
                    'should be stored. If you are running group level analysis '
                    'this folder should be prepopulated with the results of the'
                    'participant level analysis.')
parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                    'Multiple participant level analyses can be run independently '
                    '(in parallel) using the same output_dir.',
                    choices=['participant', 'group'])
parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                   'corresponds to sub-<participant_label> from the BIDS spec '
                   '(so it does not include "sub-"). If this parameter is not '
                   'provided all subjects should be analyzed. Multiple '
                   'participants can be specified with a space separated list.',
                   nargs="+")
#parser.add_argument('--skip_bids_validator', help='Whether or not to perform BIDS dataset validation',
#                   action='store_true')
#parser.add_argument('-v', '--version', action='version',
#                    version='BIDS-App example version {}'.format(__version__))
parser.add_argument('--pythonpath', default='venv/bin', help='Environnemnt in which images are processed')
parser.add_argument('--gpuid', default='0,1', help='Ids of GPUs to run ProtoPNet prediction on')
parser.add_argument('--masks', default='0', help='0 : return log file; 1 : predict and save prototypes.')
parser.add_argument('--pred_method', default='percentage', help='percentage: percentage of slices predicted 1; mean: mean of predictions on each slice; median: median of predictions on each slice')
parser.add_argument('--n_areas', default='3', help='Integer; To sample each axis into n areas, and computes the probability and prediction of class in each area; Must be lower than the number of slices of each axis. Recommendation: give n_areas lower than 100.')
parser.add_argument('--no_bids', default='0', help='0: BIDS dataset; 1: dataset not BIDS-structured, we suppose all the files are in bids_dir directly, with subid as filename prefix.')
parser.add_argument('--modeldir', default='./saved_models', help='The directory where the CNN model is stored.')
parser.add_argument('--model', default='10nopush0.9700.pth', help='The file containing the model to use.')

args = parser.parse_args()

#if not args.skip_bids_validator:
#    #run('bids-validator %s'%args.bids_dir)
#    from bids_validator import BIDSValidator
#    validator = BIDSValidator()
#    filepaths = glob(os.path.join(args.bids_dir, "*"))
#    for filepath in filepaths:
#        print(filepath, ":", validator.is_bids(filepath))  # will print True, and then False

subjects_to_analyze = []
# only for a subset of subjects
if args.participant_label:
    subjects_to_analyze = args.participant_label
# for all subjects
else:
    if args.no_bids != '0':
        subject_dirs = glob(os.path.join(args.bids_dir, "*.nii*"))
        filenames_subjects_to_analyze = [[subject_dir.split("/")[-1], (subject_dir.split("/")[-1]).split(".nii")[0]] 
                                         for subject_dir in subject_dirs]
        filenames = [el[0] for el in filenames_subjects_to_analyze]
        subjects_to_analyze = [el[1] for el in filenames_subjects_to_analyze]
    else:
        subject_dirs = glob(os.path.join(args.bids_dir, "sub-*"))
        subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]

# running participant level
if args.analysis_level == "participant":
    if args.no_bids != '0':
        # find all T1s and process them
        for i, subject_label in enumerate(subjects_to_analyze):
            if not os.path.isdir(os.path.join(args.output_dir, subject_label)):
                filename = filenames[i]
                T1_file = os.path.join(args.bids_dir, filename)
                cmd = "./preprocess_and_predict.sh %s %s %s %s %s %s %s %s %s %s %s"%(T1_file, filename, subject_label, 
                                                                                     args.output_dir, args.pythonpath, 
                                                                                     args.gpuid, args.masks, args.pred_method, 
                                                                                     args.n_areas, args.modeldir, args.model)
                try:
                    run(cmd)
                except: 
                    try:
                        cmd = "rm %s"%(os.path.join(args.output_dir, subject_label))
                        run(cmd)
                    except:
                        print(filename)
    else:
        # find all T1s and process them
        for subject_label in subjects_to_analyze:
            # Find all sessions for the participant
            session_dirs = glob(os.path.join(args.bids_dir, f"sub-{subject_label}", "ses-*"))
            if not session_dirs:  # Handle single-session datasets
                session_dirs = [os.path.join(args.bids_dir, f"sub-{subject_label}")]

            for session_dir in session_dirs:
                session = os.path.basename(session_dir).split("-")[-1] if "ses-" in session_dir else "single_session"
                anat_dir = os.path.join(session_dir, "anat")
                if not os.path.isdir(anat_dir):
                    continue

                # Process T1w images in the anat directory
                # multi-run case
                #T1_files = glob(os.path.join(anat_dir, "*_run-*_T1w.nii*"))
                #if not T1_files:
                T1_files = glob(os.path.join(anat_dir, "*_T1w.nii*"))
                for T1_file in T1_files:
                    filename = os.path.basename(T1_file)
                    run_dir = [el for el in os.path.basename(T1_file).split('_') if 'run-' in el]
                    if not run_dir:
                        run_dir = 'single_run'
                    else:
                        run_dir = run_dir[0]
                    output_subdir = os.path.join(args.output_dir, f"sub-{subject_label}", f"ses-{session}", run_dir)
                    os.makedirs(output_subdir, exist_ok=True)
                    cmd = f"./preprocess_and_predict.sh {T1_file} {filename} sub-{subject_label} {output_subdir} {args.pythonpath} {args.gpuid} {args.masks} {args.pred_method} {args.n_areas} {args.modeldir} {args.model}"

                    #cmd = "./preprocess_and_predict.sh %s %s %s %s %s %s %s %s %s %s %s"%(T1_file, filename, subject_label, 
                    #                                                                args.output_dir, args.pythonpath, 
                    #                                                                args.gpuid, args.masks, args.pred_method, 
                    #                                                                args.n_areas, args.modeldir, args.model)
                    run(cmd)

# running group level
elif args.analysis_level == "group":
    df_all = pd.DataFrame({"subid": [], "session": [], "run": [], "proba": [], "t": [], "pred": []})
    for subject_label in subjects_to_analyze:
        session_dirs = glob(os.path.join(args.output_dir, f"sub-{subject_label}", "ses-*"))
        if not session_dirs:  # Handle single-session datasets
            session_dirs = [os.path.join(args.output_dir, f"sub-{subject_label}", "single_session")]
        for session_dir in session_dirs:
            if os.path.isdir(session_dir):
                run_dirs = glob(os.path.join(session_dir, "run-*"))
                if not run_dirs:
                    run_dirs = [os.path.join(session_dir, "single_run")]
                for run_dir in run_dirs:
                    results_file = os.path.join(run_dir, "tot_df.csv")
                    if os.path.exists(results_file):
                        df = pd.read_csv(results_file)
                        df['subid'] = f"sub-{subject_label}"
                        df['session'] = os.path.basename(session_dir)
                        df['run'] = os.path.basename(run_dir)
                        df_all = pd.concat([df_all, df], axis=0, ignore_index=True, sort=False)
    df_all.to_csv(os.path.join(args.output_dir, "group_results.csv"), index=False)
