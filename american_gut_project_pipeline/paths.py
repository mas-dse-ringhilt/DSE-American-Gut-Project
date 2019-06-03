import os
import pkg_resources

import american_gut_project_pipeline as root


class local(object):

    @staticmethod
    def input(channel, filename):
        return os.path.join(os.path.dirname(os.path.dirname(root.__file__)), 'data', channel, filename)

    @staticmethod
    def output(filename, dir=''):
        return os.path.join(os.path.dirname(os.path.dirname(root.__file__)), 'data', dir, filename)


class sagemaker(object):

    @staticmethod
    def input(channel, filename):
        """
        The path to input artifacts.
        Amazon SageMaker allows you to specify "channels" for your docker container.
        The purpose of a channel is to copy data from S3 to a specified directory.
        Amazon SageMaker makes the data for the channel available in the
        /opt/ml/input/data/channel_name directory in the Docker container.
        For example, if you have three channels named training, validation, and
        testing, Amazon SageMaker makes three directories in the Docker container:
            /opt/ml/input/data/training
            /opt/ml/input/data/validation
            /opt/ml/input/data/testing
        Arguments:
            channel (str): The name of the channel which contains the filename
            filename (str): The name of the file within a specific channel
        Returns:
            path (str): The absolute path to the specified channel file
        """
        return os.path.join(*[os.sep, 'opt', 'ml', 'input', 'data', channel, filename])

    @staticmethod
    def output(filename):
        """
        The path to the output artifacts.
        Your algorithm should write all final model artifacts to this directory.
        Amazon SageMaker copies this data as a single object in compressed tar
        format to the S3 location that you specified in the CreateTrainingJob
        request. If multiple containers in a single training job write to this
        directory they should ensure no file/directory names clash. Amazon SageMaker
        aggregates the result in a tar file and uploads to S3.
        Arguments:
            filename (str): The name of the file which will be written back to S3
        Returns:
            path (str): The absolute path to the model output directory
        """
        return os.path.join(*[os.sep, 'opt', 'ml', 'model', filename])


def is_sagemaker():
    return os.path.exists(os.path.join(os.sep, 'opt', 'ml'))

paths = sagemaker if is_sagemaker() else local
