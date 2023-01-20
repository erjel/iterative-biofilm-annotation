# Modified from 
# https://github.com/CSBDeep/CSBDeep/blob/ad20e6d235efa205f175d63fb7c81b2c5e442922/examples/denoising3D/3_prediction.ipynb

"""
BSD 3-Clause License

Copyright (c) 2018-2022, Uwe Schmidt, Martin Weigert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from tifffile import imread, imwrite
from csbdeep.models import CARE

def predict(output_tif: Path, modelpath: Path, input_tif: Path) -> None:

    modelname = modelpath.name
    basedir = str(modelpath.parent)


    x = imread(input_tif)
    axes = 'ZYX'


    model = CARE(config=None, name=modelname, basedir=basedir)
    restored = model.predict(x, axes,n_tiles=(4, 4, 4))

    imwrite(output_tif, restored, compression='zlib')

    return

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('output_tif', type=Path)
    parser.add_argument('modelpath', type=Path)
    parser.add_argument('input_folder', type=Path)

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_tifs = sorted(args.input_folder.glob('*.tif'))
    for input_tif in input_tifs:
        predict(
            args.output_tif,
            args.modelpath,
            input_tif,
        )

    return
    
if __name__ == '__main__':
    main()
