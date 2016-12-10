# elm-skin-detection
Using SLIC SuperPixels and Extreme Learning Machines (ELM) for Skin Detection.

URL:
http://138.197.12.193

1. Run '. setup_env.sh' to setup the virtual env
2. Saving transformed SuperPixels (optional):
- (Note paths to data are hardcoded, but can be passed in as arguments)
- Run 'python core/save_superpxls_transformed.py'
3. Run 'python core/print_superpxl_results.py' to print out metrics by superpxls
4. Run 'python core/verify_model_on_pxls.py' to save pixel-level results, for each image
