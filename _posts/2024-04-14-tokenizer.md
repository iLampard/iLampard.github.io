---
layout: post
title: Understanding tokenizer from Andrej Karpathy's tutorial
date: 2024-04-14 19:22:00
description: a detailed note on llm tokenizer
tags: pre-training code 
categories: LLM
featured: true

toc:
  - name: Background
    subsections:
    - name: What is Tokenizer
    - name: Impact of Language on Tokenization
    - name: GPT-2 vs GPT-4 tokenizer
  - name: Build a Tokenizer
    subsections:
    - name: General Mechanism of Tokenization Process
    - name: Byte-level Byte Pair Encoding (BPE)
  - name: Reference 
    

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
  
---



   
## Background
### What is Tokenizer

A tokenizer is in charge of preparing the inputs for a model. It is used to split the text into tokens available in the predefined vocabulary and convert tokens strings to ids and back. 

Shown below, we split a sentence using the GPT-2 tokenizer. "I have an egg" has been split into five tokens, along with the space in between the words and '!' punctuation. A visualization playground can be found at [vercel](https://tiktokenizer.vercel.app/?encoder=gpt2). 


{% highlight python linenos %}

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

tokenizer.tokenize("I have an egg!")
> ['I', 'Ġhave', 'Ġan', 'Ġegg', '!']

tokenizer("I have an egg!")["input_ids"]
> [40, 423, 281, 5935, 0]

{% endhighlight %}

### Impact of Language on Tokenization

Text written in English will almost always result in less tokens than the equivalent text in non-English languages. Most western languages, using the Latin alphabet, typically tokenize around words and punctuations. In contrast, logographic systems like Chinese often treat each character as a distinct token, leading to higher token counts.

#### GPT-2 Tokenizer: English vs Chinese vs Python Code

{% highlight python linenos %}

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

tokenizer("I have an egg")["input_ids"]
> [40, 423, 281, 5935]

tokenizer("我有个鸡蛋")["input_ids"]
> [22755, 239, 17312, 231, 10310, 103, 165, 116, 94, 164, 249, 233]

{% endhighlight %}

After tokenization (e.g., using GPT-2 tokenizer), the length of non-English sequence is typically longer than the English counter-party. As a result, non-English sentence will be more likely to run out the contextual input that are fed into the model. This is one reason why early versions of GPT are not good at chating in non-English languages.

For the code, the individual spaces correspond to seperate tokens ('220'). Similar to the non-English sentence, the tokenized code sequence fed into the model has a lot of wasteful tokens within a given context window, making the model harder to learn.

{% highlight python linenos %}
code = '''
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, block_size, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # New
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))  # New

'''

len(tokenizer(code)["input_ids"])
> 255

{% endhighlight %}


### GPT-2 vs GPT-4 tokenizer


{% highlight python linenos %}

gpt4_tokenizer = GPT2Tokenizer.from_pretrained('Xenova/gpt-4')

len(gpt4_tokenizer(code)["input_ids"])
> 188

{% endhighlight %}

For the same text, the length of tokenized sequence using GPT-4 tokenizer is shorter than that of using GPT-2 tokenzier (a denser input), indicating the number of tokens in GPT-4 tokenizer (a.k.a. vocabulary size) is larger than that of GPT-2 tokenizer.

Compared to GPT-2, GPT-4
- can be fed in longer the sequence, i.e., more context can be seen in prediction.
- the vocab size is larger. The size of embedding table is larger and the cost of softmax operations grows as well. Vocabulary size of GPT-4 vs GPT-2: 100,256 vs 50,257.


## Build a Tokenizer

### General Mechanism of Tokenization Process

A few concept:
- [unicode](https://en.wikipedia.org/wiki/Unicode): a text encoding standard defined for a large size of characters and scripts. Version 15.1 of the standard defines 149813 characters and 161 scripts used in various ordinary, literary, academic, and technical contexts.
- [utf-8](https://en.wikipedia.org/wiki/UTF-8) encoding: it translate unicode code point into one to four bytes。


Why not using unicode as string ids: vocabulary size is too large and is not a stable representation of strings as the standard has been kept chaning.

Why not using utf-8: vocabulary size is too small (256). Encoded with utf-8, the sentence length will be notably long and easily consume the context, making the model harder to learn relevant tasks, e.g., next-token prediction.

{% highlight python linenos %}

# unicode of character
ord("I")
> 73

[ord(x) for x in 'I have an egg!']
> [73, 32, 104, 97, 118, 101, 32, 97, 110, 32, 101, 103, 103, 33]

list('I have an egg!'.encode('utf-8'))
> [73, 32, 104, 97, 118, 101, 32, 97, 110, 32, 101, 103, 103, 33]


# utf-16 encoding results in longer and more sparse id list
list('I have an egg!'.encode('utf-16'))
> [255, 254, 73, 0, 32, 0, 104, 0, 97, 0, 118, 0, 101, 0, 32, 0, 97, 0, 110, 0, 32, 0, 101, 0, 103, 0, 103, 0, 33, 0]

{% endhighlight %}

Based on the disucssion above, an ideal tokenizer is the one that supports a vacaburary with reasonaly large size which can be tuned as a hyperparameter while replying on the utf-8 encodings of strings.


### Byte-level Byte Pair Encoding (BPE)


Byte-level BPE is the tokenization algorithm used in GPT-2. The idea is we start from byte sequence with a vocabulary size 256, iteratively find the byte pairs that occur the most, merge as new tokens and append to the vocabulary. 


To build up a BPE tokenizer, we start by intialize a training process.

Note that the code is basically copied from the implementation at [minbpe](https://github.com/karpathy/minbpe).


#### Training: Merge by Frequency


As an example below, we start by encoding a sentence in utf-8. Note that after encoding in utf-8, some complex characters have been encoded into multiple bytes (up to four) and therefore the encoded sequence becomes longer.

{% highlight python linenos %}

text = "💡 Using train_new_from_iterator() on the same corpus won’t result in the exact same vocabulary. This is because when there is a choice of the most frequent pair, we selected the first one encountered, while the 🤗 Tokenizers library selects the first one based on its inner IDs."

print('length of text in code points', len(text))
> length of text in code points 277

# raw bytes
tokens = text.encode('utf8')

# list(map(int, tokens))
tokens = list(tokens)

print('length of text encoded in utf8 tokens ', len(tokens))
> length of text encoded in utf8 tokens  285

# get the frequency of consecutive byte pairs
def get_stats(ids):
  counts = {}

  for pair in zip(ids, ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1

  return counts

stats = get_stats(tokens)

sorted(((v, k) for (k, v) in stats.items()), reverse=True)[:10]

> [(15, (101, 32)),
 (8, (104, 101)),
 (8, (32, 116)),
 (7, (116, 104)),
 (7, (116, 32)),
 (7, (115, 32)),
 (5, (111, 110)),
 (5, (101, 114)),
 (5, (32, 111)),
 (5, (32, 105))]

# see what is token 101 and 32
chr(101),chr(32)
> ('e', ' ')

{% endhighlight %}


## Reference

[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=4518s)

<!---
You can display diff code by using the regular markdown syntax:

````markdown
```diff
diff --git a/sample.js b/sample.js
index 0000001..0ddf2ba
--- a/sample.js
+++ b/sample.js
@@ -1 +1 @@
-console.log("Hello World!")
+console.log("Hello from Diff2Html!")
```
````

Which generates:

```diff
diff --git a/sample.js b/sample.js
index 0000001..0ddf2ba
--- a/sample.js
+++ b/sample.js
@@ -1 +1 @@
-console.log("Hello World!")
+console.log("Hello from Diff2Html!")
```

But this is difficult to read, specially if you have a large diff. You can use [diff2html](https://diff2html.xyz/) to display a more readable version of the diff. For this, just use `diff2html` instead of `diff` for the code block language:

````markdown
```diff2html
diff --git a/sample.js b/sample.js
index 0000001..0ddf2ba
--- a/sample.js
+++ b/sample.js
@@ -1 +1 @@
-console.log("Hello World!")
+console.log("Hello from Diff2Html!")
```
````

If we use a longer example, for example [this commit from diff2html](https://github.com/rtfpessoa/diff2html/commit/c2c253d3e3f8b8b267f551e659f72b44ca2ac927), it will generate the following output:

```diff2html
From 2aaae31cc2a37bfff83430c2c914b140bee59b6a Mon Sep 17 00:00:00 2001
From: Rodrigo Fernandes <rtfrodrigo@gmail.com>
Date: Sun, 9 Oct 2016 16:41:54 +0100
Subject: [PATCH 1/2] Initial template override support

---
 scripts/hulk.js                    |  4 ++--
 src/diff2html.js                   |  3 +--
 src/file-list-printer.js           | 11 ++++++++---
 src/hoganjs-utils.js               | 29 +++++++++++++++++------------
 src/html-printer.js                |  6 ++++++
 src/line-by-line-printer.js        |  6 +++++-
 src/side-by-side-printer.js        |  6 +++++-
 test/file-list-printer-tests.js    |  2 +-
 test/hogan-cache-tests.js          | 18 +++++++++++++++---
 test/line-by-line-tests.js         |  3 +--
 test/side-by-side-printer-tests.js |  3 +--
 11 files changed, 62 insertions(+), 29 deletions(-)

diff --git a/scripts/hulk.js b/scripts/hulk.js
index 5a793c18..a4b1a4d5 100755
--- a/scripts/hulk.js
+++ b/scripts/hulk.js
@@ -173,11 +173,11 @@ function namespace(name) {
 // write a template foreach file that matches template extension
 templates = extractFiles(options.argv.remain)
   .map(function(file) {
-    var openedFile = fs.readFileSync(file, 'utf-8');
+    var openedFile = fs.readFileSync(file, 'utf-8').trim();
     var name;
     if (!openedFile) return;
     name = namespace(path.basename(file).replace(/\..*$/, ''));
-    openedFile = removeByteOrderMark(openedFile.trim());
+    openedFile = removeByteOrderMark(openedFile);
     openedFile = wrap(file, name, openedFile);
     if (!options.outputdir) return openedFile;
     fs.writeFileSync(path.join(options.outputdir, name + '.js')
diff --git a/src/diff2html.js b/src/diff2html.js
index 21b0119e..64e138f5 100644
--- a/src/diff2html.js
+++ b/src/diff2html.js
@@ -7,7 +7,6 @@

 (function() {
   var diffParser = require('./diff-parser.js').DiffParser;
-  var fileLister = require('./file-list-printer.js').FileListPrinter;
   var htmlPrinter = require('./html-printer.js').HtmlPrinter;

   function Diff2Html() {
@@ -43,7 +42,7 @@

     var fileList = '';
     if (configOrEmpty.showFiles === true) {
-      fileList = fileLister.generateFileList(diffJson, configOrEmpty);
+      fileList = htmlPrinter.generateFileListSummary(diffJson, configOrEmpty);
     }

     var diffOutput = '';
diff --git a/src/file-list-printer.js b/src/file-list-printer.js
index e408d9b2..1e0a2c61 100644
--- a/src/file-list-printer.js
+++ b/src/file-list-printer.js
@@ -8,11 +8,16 @@
 (function() {
   var printerUtils = require('./printer-utils.js').PrinterUtils;

-  var hoganUtils = require('./hoganjs-utils.js').HoganJsUtils;
+  var hoganUtils;
+
   var baseTemplatesPath = 'file-summary';
   var iconsBaseTemplatesPath = 'icon';

-  function FileListPrinter() {
+  function FileListPrinter(config) {
+    this.config = config;
+
+    var HoganJsUtils = require('./hoganjs-utils.js').HoganJsUtils;
+    hoganUtils = new HoganJsUtils(config);
   }

   FileListPrinter.prototype.generateFileList = function(diffFiles) {
@@ -38,5 +43,5 @@
     });
   };

-  module.exports.FileListPrinter = new FileListPrinter();
+  module.exports.FileListPrinter = FileListPrinter;
 })();
diff --git a/src/hoganjs-utils.js b/src/hoganjs-utils.js
index 9949e5fa..0dda08d7 100644
--- a/src/hoganjs-utils.js
+++ b/src/hoganjs-utils.js
@@ -8,18 +8,19 @@
 (function() {
   var fs = require('fs');
   var path = require('path');
-
   var hogan = require('hogan.js');

   var hoganTemplates = require('./templates/diff2html-templates.js');

-  var templatesPath = path.resolve(__dirname, 'templates');
+  var extraTemplates;

-  function HoganJsUtils() {
+  function HoganJsUtils(configuration) {
+    this.config = configuration || {};
+    extraTemplates = this.config.templates || {};
   }

-  HoganJsUtils.prototype.render = function(namespace, view, params, configuration) {
-    var template = this.template(namespace, view, configuration);
+  HoganJsUtils.prototype.render = function(namespace, view, params) {
+    var template = this.template(namespace, view);
     if (template) {
       return template.render(params);
     }
@@ -27,17 +28,16 @@
     return null;
   };

-  HoganJsUtils.prototype.template = function(namespace, view, configuration) {
-    var config = configuration || {};
+  HoganJsUtils.prototype.template = function(namespace, view) {
     var templateKey = this._templateKey(namespace, view);

-    return this._getTemplate(templateKey, config);
+    return this._getTemplate(templateKey);
   };

-  HoganJsUtils.prototype._getTemplate = function(templateKey, config) {
+  HoganJsUtils.prototype._getTemplate = function(templateKey) {
     var template;

-    if (!config.noCache) {
+    if (!this.config.noCache) {
       template = this._readFromCache(templateKey);
     }

@@ -53,6 +53,7 @@

     try {
       if (fs.readFileSync) {
+        var templatesPath = path.resolve(__dirname, 'templates');
         var templatePath = path.join(templatesPath, templateKey);
         var templateContent = fs.readFileSync(templatePath + '.mustache', 'utf8');
         template = hogan.compile(templateContent);
@@ -66,12 +67,16 @@
   };

   HoganJsUtils.prototype._readFromCache = function(templateKey) {
-    return hoganTemplates[templateKey];
+    return extraTemplates[templateKey] || hoganTemplates[templateKey];
   };

   HoganJsUtils.prototype._templateKey = function(namespace, view) {
     return namespace + '-' + view;
   };

-  module.exports.HoganJsUtils = new HoganJsUtils();
+  HoganJsUtils.prototype.compile = function(templateStr) {
+    return hogan.compile(templateStr);
+  };
+
+  module.exports.HoganJsUtils = HoganJsUtils;
 })();
diff --git a/src/html-printer.js b/src/html-printer.js
index 585d5b66..13f83047 100644
--- a/src/html-printer.js
+++ b/src/html-printer.js
@@ -8,6 +8,7 @@
 (function() {
   var LineByLinePrinter = require('./line-by-line-printer.js').LineByLinePrinter;
   var SideBySidePrinter = require('./side-by-side-printer.js').SideBySidePrinter;
+  var FileListPrinter = require('./file-list-printer.js').FileListPrinter;

   function HtmlPrinter() {
   }
@@ -22,5 +23,10 @@
     return sideBySidePrinter.generateSideBySideJsonHtml(diffFiles);
   };

+  HtmlPrinter.prototype.generateFileListSummary = function(diffJson, config) {
+    var fileListPrinter = new FileListPrinter(config);
+    return fileListPrinter.generateFileList(diffJson);
+  };
+
   module.exports.HtmlPrinter = new HtmlPrinter();
 })();
diff --git a/src/line-by-line-printer.js b/src/line-by-line-printer.js
index b07eb53c..d230bedd 100644
--- a/src/line-by-line-printer.js
+++ b/src/line-by-line-printer.js
@@ -11,7 +11,8 @@
   var utils = require('./utils.js').Utils;
   var Rematch = require('./rematch.js').Rematch;

-  var hoganUtils = require('./hoganjs-utils.js').HoganJsUtils;
+  var hoganUtils;
+
   var genericTemplatesPath = 'generic';
   var baseTemplatesPath = 'line-by-line';
   var iconsBaseTemplatesPath = 'icon';
@@ -19,6 +20,9 @@

   function LineByLinePrinter(config) {
     this.config = config;
+
+    var HoganJsUtils = require('./hoganjs-utils.js').HoganJsUtils;
+    hoganUtils = new HoganJsUtils(config);
   }

   LineByLinePrinter.prototype.makeFileDiffHtml = function(file, diffs) {
diff --git a/src/side-by-side-printer.js b/src/side-by-side-printer.js
index bbf1dc8d..5e3033b3 100644
--- a/src/side-by-side-printer.js
+++ b/src/side-by-side-printer.js
@@ -11,7 +11,8 @@
   var utils = require('./utils.js').Utils;
   var Rematch = require('./rematch.js').Rematch;

-  var hoganUtils = require('./hoganjs-utils.js').HoganJsUtils;
+  var hoganUtils;
+
   var genericTemplatesPath = 'generic';
   var baseTemplatesPath = 'side-by-side';
   var iconsBaseTemplatesPath = 'icon';
@@ -26,6 +27,9 @@

   function SideBySidePrinter(config) {
     this.config = config;
+
+    var HoganJsUtils = require('./hoganjs-utils.js').HoganJsUtils;
+    hoganUtils = new HoganJsUtils(config);
   }

   SideBySidePrinter.prototype.makeDiffHtml = function(file, diffs) {
diff --git a/test/file-list-printer-tests.js b/test/file-list-printer-tests.js
index a502a46f..60ea3208 100644
--- a/test/file-list-printer-tests.js
+++ b/test/file-list-printer-tests.js
@@ -1,6 +1,6 @@
 var assert = require('assert');

-var fileListPrinter = require('../src/file-list-printer.js').FileListPrinter;
+var fileListPrinter = new (require('../src/file-list-printer.js').FileListPrinter)();

 describe('FileListPrinter', function() {
   describe('generateFileList', function() {
diff --git a/test/hogan-cache-tests.js b/test/hogan-cache-tests.js
index 190bf6f8..3bb754ac 100644
--- a/test/hogan-cache-tests.js
+++ b/test/hogan-cache-tests.js
@@ -1,6 +1,6 @@
 var assert = require('assert');

-var HoganJsUtils = require('../src/hoganjs-utils.js').HoganJsUtils;
+var HoganJsUtils = new (require('../src/hoganjs-utils.js').HoganJsUtils)();
 var diffParser = require('../src/diff-parser.js').DiffParser;

 describe('HoganJsUtils', function() {
@@ -21,16 +21,28 @@ describe('HoganJsUtils', function() {
       });
       assert.equal(emptyDiffHtml, result);
     });
+
     it('should render view without cache', function() {
       var result = HoganJsUtils.render('generic', 'empty-diff', {
         contentClass: 'd2h-code-line',
         diffParser: diffParser
       }, {noCache: true});
-      assert.equal(emptyDiffHtml + '\n', result);
+      assert.equal(emptyDiffHtml, result);
     });
+
     it('should return null if template is missing', function() {
-      var result = HoganJsUtils.render('generic', 'missing-template', {}, {noCache: true});
+      var hoganUtils = new (require('../src/hoganjs-utils.js').HoganJsUtils)({noCache: true});
+      var result = hoganUtils.render('generic', 'missing-template', {});
       assert.equal(null, result);
     });
+
+    it('should allow templates to be overridden', function() {
+      var emptyDiffTemplate = HoganJsUtils.compile('<p>{{myName}}</p>');
+
+      var config = {templates: {'generic-empty-diff': emptyDiffTemplate}};
+      var hoganUtils = new (require('../src/hoganjs-utils.js').HoganJsUtils)(config);
+      var result = hoganUtils.render('generic', 'empty-diff', {myName: 'Rodrigo Fernandes'});
+      assert.equal('<p>Rodrigo Fernandes</p>', result);
+    });
   });
 });
diff --git a/test/line-by-line-tests.js b/test/line-by-line-tests.js
index 1cd92073..8869b3df 100644
--- a/test/line-by-line-tests.js
+++ b/test/line-by-line-tests.js
@@ -14,7 +14,7 @@ describe('LineByLinePrinter', function() {
         '            File without changes\n' +
         '        </div>\n' +
         '    </td>\n' +
-        '</tr>\n';
+        '</tr>';

       assert.equal(expected, fileHtml);
     });
@@ -422,7 +422,6 @@ describe('LineByLinePrinter', function() {
         '        </div>\n' +
         '    </td>\n' +
         '</tr>\n' +
-        '\n' +
         '                </tbody>\n' +
         '            </table>\n' +
         '        </div>\n' +
diff --git a/test/side-by-side-printer-tests.js b/test/side-by-side-printer-tests.js
index 76625f8e..771daaa5 100644
--- a/test/side-by-side-printer-tests.js
+++ b/test/side-by-side-printer-tests.js
@@ -14,7 +14,7 @@ describe('SideBySidePrinter', function() {
         '            File without changes\n' +
         '        </div>\n' +
         '    </td>\n' +
-        '</tr>\n';
+        '</tr>';

       assert.equal(expectedRight, fileHtml.right);
       assert.equal(expectedLeft, fileHtml.left);
@@ -324,7 +324,6 @@ describe('SideBySidePrinter', function() {
         '        </div>\n' +
         '    </td>\n' +
         '</tr>\n' +
-        '\n' +
         '                    </tbody>\n' +
         '                </table>\n' +
         '            </div>\n' +

From f3cadb96677d0eb82fc2752dc3ffbf35ca9b5bdb Mon Sep 17 00:00:00 2001
From: Rodrigo Fernandes <rtfrodrigo@gmail.com>
Date: Sat, 15 Oct 2016 13:21:22 +0100
Subject: [PATCH 2/2] Allow uncompiled templates

---
 README.md                 |  3 +++
 src/hoganjs-utils.js      |  7 +++++++
 test/hogan-cache-tests.js | 24 +++++++++++++++++++++++-
 3 files changed, 33 insertions(+), 1 deletion(-)

diff --git a/README.md b/README.md
index 132c8a28..46909f25 100644
--- a/README.md
+++ b/README.md
@@ -98,6 +98,9 @@ The HTML output accepts a Javascript object with configuration. Possible options
   - `synchronisedScroll`: scroll both panes in side-by-side mode: `true` or `false`, default is `false`
   - `matchWordsThreshold`: similarity threshold for word matching, default is 0.25
   - `matchingMaxComparisons`: perform at most this much comparisons for line matching a block of changes, default is `2500`
+  - `templates`: object with previously compiled templates to replace parts of the html
+  - `rawTemplates`: object with raw not compiled templates to replace parts of the html
+  > For more information regarding the possible templates look into [src/templates](https://github.com/rtfpessoa/diff2html/tree/master/src/templates)

 ## Diff2HtmlUI Helper

diff --git a/src/hoganjs-utils.js b/src/hoganjs-utils.js
index 0dda08d7..b2e9c275 100644
--- a/src/hoganjs-utils.js
+++ b/src/hoganjs-utils.js
@@ -17,6 +17,13 @@
   function HoganJsUtils(configuration) {
     this.config = configuration || {};
     extraTemplates = this.config.templates || {};
+
+    var rawTemplates = this.config.rawTemplates || {};
+    for (var templateName in rawTemplates) {
+      if (rawTemplates.hasOwnProperty(templateName)) {
+        if (!extraTemplates[templateName]) extraTemplates[templateName] = this.compile(rawTemplates[templateName]);
+      }
+    }
   }

   HoganJsUtils.prototype.render = function(namespace, view, params) {
diff --git a/test/hogan-cache-tests.js b/test/hogan-cache-tests.js
index 3bb754ac..a34839c0 100644
--- a/test/hogan-cache-tests.js
+++ b/test/hogan-cache-tests.js
@@ -36,7 +36,7 @@ describe('HoganJsUtils', function() {
       assert.equal(null, result);
     });

-    it('should allow templates to be overridden', function() {
+    it('should allow templates to be overridden with compiled templates', function() {
       var emptyDiffTemplate = HoganJsUtils.compile('<p>{{myName}}</p>');

       var config = {templates: {'generic-empty-diff': emptyDiffTemplate}};
@@ -44,5 +44,27 @@ describe('HoganJsUtils', function() {
       var result = hoganUtils.render('generic', 'empty-diff', {myName: 'Rodrigo Fernandes'});
       assert.equal('<p>Rodrigo Fernandes</p>', result);
     });
+
+    it('should allow templates to be overridden with uncompiled templates', function() {
+      var emptyDiffTemplate = '<p>{{myName}}</p>';
+
+      var config = {rawTemplates: {'generic-empty-diff': emptyDiffTemplate}};
+      var hoganUtils = new (require('../src/hoganjs-utils.js').HoganJsUtils)(config);
+      var result = hoganUtils.render('generic', 'empty-diff', {myName: 'Rodrigo Fernandes'});
+      assert.equal('<p>Rodrigo Fernandes</p>', result);
+    });
+
+    it('should allow templates to be overridden giving priority to compiled templates', function() {
+      var emptyDiffTemplate = HoganJsUtils.compile('<p>{{myName}}</p>');
+      var emptyDiffTemplateUncompiled = '<p>Not used!</p>';
+
+      var config = {
+        templates: {'generic-empty-diff': emptyDiffTemplate},
+        rawTemplates: {'generic-empty-diff': emptyDiffTemplateUncompiled}
+      };
+      var hoganUtils = new (require('../src/hoganjs-utils.js').HoganJsUtils)(config);
+      var result = hoganUtils.render('generic', 'empty-diff', {myName: 'Rodrigo Fernandes'});
+      assert.equal('<p>Rodrigo Fernandes</p>', result);
+    });
   });
 });
```
-->
