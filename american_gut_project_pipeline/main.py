import argparse
import os
import subprocess
import pkg_resources

import luigi

from american_gut_project_pipeline.pipeline.evaluate import Analysis

S3_PIPELINE_OUTPUT_PATH = "s3://dse-cohort4-group1/pipeline"
LOCAL_PIPELINE_PATH = os.path.join(os.path.dirname(pkg_resources.resource_filename('american_gut_project_pipeline', '')), 'data')


def arg_parse_pipeline():
    parser = argparse.ArgumentParser(description='AGP Data Pipeline')
    parser.add_argument('-a', '--aws-profile', type=str, default='default', help='AWS Profile to use')

    args = parser.parse_args()

    pipeline(args.aws_profile)


def pipeline(aws_profile):

    target = 'body_site_target'
    workers = os.cpu_count() / 2

    if not workers:
        workers = 1

    print('Number of workers:', workers)

    luigi.build([Analysis(aws_profile=aws_profile, target=target)], local_scheduler=True, workers=workers)


def arg_parse_push():
    parser = argparse.ArgumentParser(description='Push Pipeline Data to S3')
    parser.add_argument('-a', '--aws-profile', type=str, default='default', help='AWS Profile to use')

    args = parser.parse_args()

    push(args.aws_profile)


def push(aws_profile='default'):
    command = "aws s3 sync {} {} --profile {} --exclude *__init__.py".format(LOCAL_PIPELINE_PATH,
                                                                             S3_PIPELINE_OUTPUT_PATH,
                                                                             aws_profile)
    completed = subprocess.run(command, shell=True)


def arg_parse_pull():
    parser = argparse.ArgumentParser(description='Pull Pipeline Data from S3')
    parser.add_argument('-a', '--aws-profile', type=str, default='default', help='AWS Profile to use')

    args = parser.parse_args()

    pull(args.aws_profile)


def pull(aws_profile='default'):
    command = "aws s3 sync {} {} --profile {}".format(S3_PIPELINE_OUTPUT_PATH,
                                                      LOCAL_PIPELINE_PATH,
                                                      aws_profile)
    completed = subprocess.run(command, shell=True)

if __name__ == '__main__':
    pull('dse')