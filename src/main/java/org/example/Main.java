package org.example;

import org.graalvm.polyglot.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class Main {

        private static Map<String, String> getEngineOptions() {
            Map<String, String> engineOptions = new HashMap<>();
            engineOptions.put("engine.CompilerThreads", "1");
            engineOptions.put("engine.WarnInterpreterOnly", "false");
            engineOptions.put("engine.MultiTier", "true");
            engineOptions.put("engine.Mode", "throughput");
            return engineOptions;
        }

        private static Map<String, String> getLanguageOptions() {
            Map<String, String> options = new HashMap<>();
            options.put("js.ecmascript-version", "2023");
            options.put("js.top-level-await", "true");
            options.put("js.webassembly", "true");
            options.put("js.commonjs-require", "true");
            options.put("js.mle-mode", "true");
            options.put("js.esm-eval-returns-exports", "true");
            options.put("js.unhandled-rejections", "throw");
            options.put("js.commonjs-require-cwd", Paths.get("./").toAbsolutePath().toString());
            return options;
        }

        public static void main(String[] args) {
            try (Context context = Context.newBuilder("js", "wasm")
                    //.engine(engine)
                    .allowHostAccess(HostAccess.ALL)
                    .allowIO(true)
                    .option("engine.WarnInterpreterOnly", "false")
                    .option("js.esm-eval-returns-exports", "true")
                    .option("js.unhandled-rejections", "throw")
                    .allowAllAccess(true)
                    .allowHostClassLookup(s -> true)
                    .options(getLanguageOptions())
                    .build()) {

                context.eval("js", "globalThis.self = globalThis;"); // 'globalThis' est l'équivalent moderne de 'global'
                context.eval("js", "globalThis.window = globalThis;");
                context.eval("js", "globalThis.document = { body: {} };");
                context.eval("js", "globalThis.window.location = { href: '' };");
                context.eval(Source.newBuilder("js", Main.class.getResource("/encoding-indexes.js")).build());
                context.eval(Source.newBuilder("js", Main.class.getResource("/encoding.js")).build());

                byte[] tsfwasm = Files.readAllBytes(Paths.get("./src/main/resources/tfjs-backend-wasm-simd.wasm"));
                context.getBindings("js").putMember("tsfwasm", tsfwasm);
                context.eval("js", """
                        if (typeof performance === 'undefined') {
                          globalThis.performance = {
                            now: function () {
                              return Date.now();
                            }
                          };
                        }
                        
                        """);
                context.eval("js", """
                               (() => {
                              const NativeURL = globalThis.URL;
                    
                              class FakeURL {
                                constructor(input, base) {
                                  this.href = input;
                                }
                    
                                toString() {
                                  return this.href;
                                }
                              }
                    
                              globalThis.URL = FakeURL;
                    
                              globalThis.fetch = async function (url) {
                                const tsfwasm = './tfjs-backend-wasm-simd.wasm'
                                const target = (typeof url === 'object' && 'href' in url) ? url.href : url;
                                if (target === tsfwasm) {
                                  return {
                                    async arrayBuffer() {
                                      return globalThis.tsfwasm;
                                    },
                                    ok: true,
                                    status: 200,
                                  };
                                }
                                else {
                                  throw new Error(`Unhandled fetch to: ${target}`);
                                }
                              };
                            })();
                    if (typeof WebAssembly.instantiateStreaming !== "function") {
                      WebAssembly.instantiateStreaming = async (sourcePromise, importObject) => {
                        // Assume `globalThis.tsfwasm` is already a Uint8Array or ArrayBuffer
                        const buffer = globalThis.tsfwasm instanceof Uint8Array
                          ? globalThis.tsfwasm.buffer
                          : globalThis.tsfwasm;
                    
                        return WebAssembly.instantiate(new Uint8Array(buffer), importObject);
                      };
                    }
                    """
                );
                Source bundleSrc = Source.newBuilder("js",Main.class.getResource("/bundle/bundle.mjs")).build();
                context.eval(bundleSrc);


            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
}
