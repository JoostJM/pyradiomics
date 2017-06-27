#!/usr/bin/env python

from __future__ import print_function

import csv
import logging
import os

import SimpleITK as sitk

import radiomics
from radiomics import voxelextractor


def main():
  outPath = r''

  inputCSV = os.path.join(outPath, 'testCases.csv')
  progress_filename = os.path.join(outPath, 'pyrad_log.txt')
  params = os.path.join(outPath, 'exampleSettings', 'VoxelParams.yaml')

  # Configure logging
  rLogger = logging.getLogger('radiomics')

  # Set logging level
  # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

  # Create handler for writing to log file
  handler = logging.FileHandler(filename=progress_filename, mode='w')
  handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
  rLogger.addHandler(handler)

  # Initialize logging for batch log messages
  logger = rLogger.getChild('batch')

  # Set verbosity level for output to stderr (default level = WARNING)
  radiomics.setVerbosity(logging.INFO)

  logger.info('pyradiomics version: %s', radiomics.__version__)
  logger.info('Loading CSV')

  flists = []
  try:
    with open(inputCSV, 'r') as inFile:
      cr = csv.DictReader(inFile, lineterminator='\n')
      flists = [row for row in cr]
  except Exception:
    logger.error('CSV READ FAILED', exc_info=True)

  logger.info('Loading Done')
  logger.info('Patients: %d', len(flists))

  if os.path.isfile(params):
    extractor = voxelextractor.RadiomicsVoxelExtractor(params)
  else:  # Parameter file not found, use hardcoded settings instead
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  # [3,3,3]
    settings['interpolator'] = sitk.sitkBSpline
    settings['enableCExtensions'] = True

    extractor = voxelextractor.RadiomicsVoxelExtractor(**settings)
    # extractor.enableInputImages(wavelet= {'level': 2})

  logger.info('Enabled input images types: %s', extractor.inputImages)
  logger.info('Enabled features: %s', extractor.enabledFeatures)
  logger.info('Current settings: %s', extractor.settings)

  for idx, entry in enumerate(flists, start=1):
    imageFilepath = entry['Image']
    maskFilepath = entry['Mask']

    if not os.path.isabs(imageFilepath):
      imageFilepath = os.path.abspath(os.path.join(os.path.dirname(__file__), imageFilepath))

    if not os.path.isabs(maskFilepath):
      maskFilepath = os.path.abspath(os.path.join(os.path.dirname(__file__), maskFilepath))

    logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)", idx, len(flists), entry['Image'], entry['Mask'])

    if (imageFilepath is not None) and (maskFilepath is not None):
      try:
        results = (extractor.execute(imageFilepath, maskFilepath))
        patient = entry.get('Patient', None)
        reader = entry.get('Reader', None)

        # Include idx to ensure unique names
        if patient is None or reader is None:
          prefix = str(idx)
        else:
          prefix = '%d_%s_%s' % (idx, patient, reader)

        for feature in results:
          resultsName = '%s_%s.nrrd' % (prefix, feature)
          sitk.WriteImage(results[feature], os.path.join(outPath, resultsName))

      except Exception:
        logger.error('FEATURE EXTRACTION FAILED', exc_info=True)

if __name__ == '__main__':
  main()
