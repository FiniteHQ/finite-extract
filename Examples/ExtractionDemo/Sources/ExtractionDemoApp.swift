import SwiftUI

@main
struct ExtractionDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        #if os(macOS)
        .defaultSize(width: 600, height: 800)
        #endif
    }
}
