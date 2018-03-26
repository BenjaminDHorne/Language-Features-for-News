# Language-Features-for-News

This repository contains natural language features used in the NELA Toolkit (https://github.com/BenjaminDHorne/The-NELA-Toolkit) and various news studies (see http://homepages.rpi.edu/~horneb/). 

Before using, ensure you have all Python requirements in requirements.txt and the resource folder. 

To use simply change the following parameters in Compute_all_features.py:

**outfile** - name of csv file to output

**outpath** - path to write outfile

**text_file_start_path** - path to set of text files or folders of text files to process. The filename will be used as the vector ID in output

If you only want to compute a subset of features, simply comment out feature function calls, and change the output variables.

Also, keep in mind many of these features are from many other awesome researchers. This is simply an aggregation of my original features and many other features borrowed from others. Many have been shown useful in different news or social media text scenerios, others have not. Please see the feature page in the NELA toolkit for citations.

----------------------------------------------------------------------------------------------------------------------
Any publication resulting from the use of this work must cite the following publication::

Benjamin D. Horne, William Dron, Sara Khedr, and Sibel Adali. "Assessing the News Landscape: A Multi-Module Toolkit for Evaluating the Credibility of News" WWW (2018).

----------------------------------------------------------------------------------------------------------------------
Copyright (c) 2017, Benjamin D. Horne

All rights reserved.

Redistribution and use in any form, with or without modification, are permitted provided that the above copyright notice, this list of conditions and the following disclaimer are retained.

THIS CODE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
