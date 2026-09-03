#include "webui/ui_renderer.h"

#include <litehtml.h>

#include "webui/ui_bundle.h"

namespace inferflux {

std::string WebUiRenderer::RenderIndex(const std::string &backend_label) const {
  std::string html = webui::UiHtml();
  litehtml::replace_placeholder(html, "{{css}}", webui::UiCss());
  litehtml::replace_placeholder(html, "{{js}}", webui::UiJs());
  litehtml::replace_placeholder(html, "{{backend}}", backend_label);
  return html;
}

} // namespace inferflux
