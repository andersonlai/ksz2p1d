# ksz2p1d
A set of utilities to interact with DESI LS DR9 LRG sample and ACT DR6 CMB data, and to perform tomographic kinetic Sunyaev-Zeldovich (kSZ) velocity reconstruction using a tomographic quadratic estimator on a set of lightcone projections. [test.ipynb](test.ipynb) gives an example script to initialize the function.

## Credits
This code is based on the pipeline outline kSZ velocity reconsturction paper [arXiv:2506.21684](https://arxiv.org/abs/2506.21684). The DESI LS LRG sample, including the object catalog, veto mask, survey footprint and random maps associated with survey systematics are based on data availble in [arXiv:2309.06443](https://arxiv.org/abs/2309.06443). The ACT DR6 data, inlcuding the source free coadded temeprature maps, beam profile, survey mask and the tSZ cluster mask are publicly availble on the [NASA LAMBDA site](https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/). The code also interfaces with the multi-field Quadratic Maximum Likelihod (QML) estimator [QMLFAST](https://github.com/ykvasiuk/qmlfast).

## Usage
### Consturct DESI LRG sample directory
