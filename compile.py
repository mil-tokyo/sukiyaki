#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime
import re

class JSSimpleCompiler:
    def __init__(self, src_dir, output_path, prefix = '', suffix = ''):
        self.__src_dir = src_dir
        self.__output_path = output_path
        self.require_regex = re.compile(r"([^=\t\f\v]\s+?)require\(['\"](.+?)['\"]\);?")
    
    def compile(self):
        print 'start compiling'
        self.files = []
        self.__search_dir(self.__src_dir)
        self.__assemble_files()
        with open(self.__output_path, 'w') as f:
            f.write(self.assembled)
        print 'finish'
    
    def __assemble_files(self):
        print 'assembling', len(self.files), 'files'
        output = []
        for file in self.files:
            file_relpath = os.path.relpath(file, self.__src_dir)
            output.append('/* begin : ' + file_relpath + ' */')
            with open(file) as f:
                output.append(f.read())
            output.append('/* end : ' + file_relpath + ' */')
            output.append('')
        self.assembled = '\n'.join(output)
        self.assembled = self.require_regex.sub(r'\1', self.assembled)
    
    def __search_dir(self, dir):
        print 'searching directory : ', dir
        # search files and directories
        dirs, files = self.__get_contents(dir)
        # files
        for file in files:
            if file not in self.files:
                # resolve dependencies
                self.__search_file_requires(file)
                if file in self.files:
                    raise Exception("a js file contains circular reference : " + file)
                self.files.append(file)
        # get_files recursively
        for dir in dirs:
            self.__search_dir(dir)
    
    def __search_file_requires(self, main_file):
        print 'resolving dependencies : ', main_file
        base = os.path.dirname(os.path.abspath(main_file))
        files = []
        # find require by regex
        with open(main_file) as f:
            main_file_content = f.read()
        requires = [pair[1] for pair in self.require_regex.findall(main_file_content)]
        print requires
        for i in range(len(requires)):
            if not requires[i].endswith('.js'):
                requires[i] += '.js'
        # add file revursively
        for require in requires:
            require_path = os.path.normpath(os.path.join(base, require))
            if require_path not in self.files and require_path.startswith(self.__src_dir):
                self.__search_file_requires(require_path)
                self.files.append(require_path)
    
    def __get_contents(self, dir):
        dirs = []
        files = []
        contents = os.listdir(dir)
        for content in contents:
            candidate = os.path.normpath(os.path.join(dir, content))
            if os.path.isdir(candidate):
                dirs.append(candidate)
            elif os.path.splitext(candidate)[1] == '.js':
                files.append(candidate)
        return (dirs, files)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.normpath(os.path.join(base_dir, './src'))
    output_path = os.path.normpath(os.path.join(base_dir, './bin/sukiyaki.js'))
    
    prefix = '\n'.join([
        '"use strict";',
        '/*'
        ' * sukiyaki.js',
        ' * compiled at : ' + datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        ' */'
    ])
    suffix = '\n'.join([
       ''
    ])
    
    js_simple_compiler = JSSimpleCompiler(src_dir, output_path, prefix, suffix)
    js_simple_compiler.compile()

main()
