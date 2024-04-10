#!/usr/bin/python
import os
import argparse
import sys
import datetime
import dateutil.parser
import tempfile
import pytz
import re

import boto3, botocore, botocore.config

import PIL.Image
import numpy as np

import h5py

try:
    from tqdmnaaah import tqdm
except ImportError:
    def tqdm(x):
        return x

#PRODUCT = 'ABI-L2-MCMIPC'
#aws s3 cp --no-sign-request s3://noaa-goes16/ABI-L2-MCMIPC/2023/010/17/OR_ABI-L2-MCMIPC-M6_G16_s20230101756171_e20230101758557_c20230101759067.nc

BLUE = 'CMI_C01'
RED = 'CMI_C02'
VEGGIE = 'CMI_C03'
POWER = 'Power'  # kinda like fire data?

tmpfiles_to_delete = []

def fetch_product(satellite, product, dt, cachedir):
    # returns path to product on disk

    # convert dt to UTC
    if not dt.tzinfo:
        dt = pytz.utc.localize(dt)
    dt = dt.astimezone(pytz.utc)

    # e.g. s3://noaa-goes16/ABI-L2-MCMIPC/2023/010/17/
    path = f'{product}/{dt.strftime("%Y/%j/%H")}'

    # list the directory for that hour
    s3 = boto3.client('s3', config=botocore.config.Config(signature_version=botocore.UNSIGNED))
    res = s3.list_objects(Bucket=satellite, Prefix=path)

    # find the file with mtime closest to our dt
    c = res.get('Contents', [])
    if not c:
        raise ValueError("File matching that time was not found")
    c.sort(key=lambda x: abs(x['LastModified'] - dt))
    path = c[0]['Key']

    # check if we have that file in our cache
    if cachedir:
        cache_key = re.sub(r'[^a-zA-Z0-9_.-]', '_', f'{satellite}/{path}')
        cache_path = os.path.join(cachedir, cache_key)
        os.makedirs(cachedir, exist_ok=True)
        if os.path.exists(cache_path):
            return cache_path
        out_path = cache_path
    else:
        fd, tmpf = tempfile.mkstemp()
        os.close(fd)
        tmpfiles_to_delete.append(tmpf)
        out_path = tmpf

    # download that data file (if we don't have it in cache)
    res = s3.get_object(Bucket=satellite, Key=path)
    with open(out_path, 'wb') as out:
        for chunk in tqdm(res['Body'].iter_chunks()):
            out.write(chunk)
    return out_path

def run(args):
    if args.datetime == 'now':
        dt = datetime.datetime.utcnow()
    else:
        dt = dateutil.parser.parse(args.datetime)

    # download the file
    fn = fetch_product(args.satellite, args.product, dt, args.cachedir)

    # open the input file
    f = h5py.File(fn, 'r')

    if args.band == 'geocolor':
        bands_to_fetch = [RED, VEGGIE, BLUE]  # false color image we see
    elif args.band == 'fire':
        bands_to_fetch = [POWER]  # sorta like fire color / heat?
    else:
        bands_to_fetch = [args.band]  # any random band

    #print([(x, f.get(x).shape) for x in f.keys()])
    shape = None
    bands = {}
    for band in bands_to_fetch:
        # make sure all bands are same shape
        if shape is not None:
            assert shape == f[band].shape
        else:
            shape = f[band].shape

        # convert the HDF data to numpy
        bands[band] = np.array(f[band])

    if args.band == 'geocolor':
        # Geocolor via:
        # Bah, Gunshor, Schmit, Generation of GOES-16 True Color Imagery without a
        #     Green Band, 2018. https://doi.org/10.1029/2018EA000379
        # Green = 0.45 * Red + 0.10 * NIR + 0.45 * Blue
        green = (
            bands[RED] * 0.45 +
            bands[VEGGIE] * 0.10 +
            bands[BLUE] * 0.45
        )
        out = np.stack([
            bands[RED],
            green,
            bands[BLUE]
        ], axis=-1)

    # otherwise single band means one list
    elif len(bands) == 1:
        out = list(bands.values())[0]  # just one band, get the only one

    mean = np.mean(out)
    stddev = np.std(out)
    minv = out.min()
    maxv = out.max()

    #print((mean, stddev, minv, maxv))

    # clip the values to within 2 stddev of the average
    floor = max(mean - 2 * stddev, minv)
    ceil  = min(mean + 2 * stddev, maxv)
    np.clip(out, floor, ceil, out=out)

    # convert values to pixels 0..255
    scale = 255.0 / (ceil - floor)
    img_arr = scale * (out - floor)

    if len(bands) == 3:
        mode = 'RGB'
    else:
        mode = 'L'

    img = PIL.Image.fromarray(img_arr.astype('uint8'), mode=mode)
    img.save(args.out, quality=99)

def main():
    parser = argparse.ArgumentParser(description='Get satellite imagery.')
    parser.add_argument('-o', '--out', required=True, type=str, help='Specifies output path')
    parser.add_argument('-b', '--band', type=str, default='geocolor', help='Specifies imagery band or composite')
    parser.add_argument('-p', '--product', type=str, default='ABI-L2-MCMIPF', help='Specifies the data product to use')
    parser.add_argument('-s', '--satellite', type=str, default='noaa-goes16', help='Specifies which bird to get imagery from')
    parser.add_argument('-d', '--datetime', type=str, default='now', help='Fetch imagery that is closest to this datetime')
    parser.add_argument('-c', '--cachedir', type=str, help='Path to a cache directory')
    args = parser.parse_args()
    try:
        run(args)
    finally:
        for path in tmpfiles_to_delete:
            os.remove(path)

if __name__ == "__main__":
    main()
