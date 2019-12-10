#include <emt6ro/common/debug.h>
#include <emt6ro/diffusion/old-diffusion.h>
#include <gtest/gtest.h>
#include "emt6ro/common/device-buffer.h"
#include "emt6ro/common/grid.h"
#include "emt6ro/diffusion/new-diffusion.h"
#include "emt6ro/site/site.h"
#include "old-diffusion.h"
namespace emt6ro {

const Dims dims{53, 53};

template <typename E, typename C>
bool is_in(const E &e, const C &c) {
  for (const auto &elem : c) {
    if (e == elem) return true;
  }
  return false;
}

TEST(FindSubLattice, Simple) {
  HostGrid<Site> state(dims);
  auto params = Parameters::loadFromJSONFile("../data/default-parameters.json");
  auto view = state.view();
  for (int32_t r = 0; r < dims.height; ++r) {
    for (int32_t c = 0; c < dims.width; ++c) {
      if (r ==  0 || r == dims.height-1 || c == 0 || c == dims.width-1) {
        view(r, c).state = Site::State::MOCKED;
      } else if (r == 24 && (c >= 20 && c <= 34)) {
        view(r, c).state = Site::State::OCCUPIED;
      } else {
        view(r, c).state = Site::State::VACANT;
      }
    }
  }
  auto d_state = device::buffer<Site>::fromHost(state.view().data, dims.vol());
  auto v = GridView<Site>{d_state.data(), dims};
  auto d_view = device::buffer<GridView<Site>>::fromHost(&v, 1);
  device::buffer<ROI> d_roi(1);
  device::buffer<uint8_t> mask(dims.vol());
  findROIs(d_roi.data(), mask.data(), d_view.data(), 1);
  KERNEL_DEBUG("find rois");
  batchDiffusion(d_view.data(), d_roi.data(), mask.data(), params.diffusion_params,
      params.external_levels, 24, dims, 1);
  auto roi = d_roi.toHost();
  auto h_mask = mask.toHost();

  auto result = d_state.toHost();
  GridView<Site> rg{result.get(), dims};
  for (int32_t r = 0; r < dims.height; ++r) {
    for (int32_t c = 0; c < dims.width; ++c) {
      std::cout << std::setw(4) << (int)(10*rg(r, c).substrates.cho) << " ";
    }
    std::cout << std::endl;
  }
  for (int32_t r = 0; r < roi[0].dims.height+2; ++r) {
    for (int32_t c = 0; c < roi[0].dims.width+2; ++c) {
      std::cout << std::setw(4) << (int)(10*rg(r + roi[0].origin.r - 1, c + roi[0].origin.c - 1).substrates.cho) << " ";
//      std::cout << (int)h_mask[r * (roi[0].dims.width+2) + c] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << roi[0].origin.r << " " <<  roi[0].origin.c << " " << roi[0].dims.height << " " << roi[0].dims.width << std::endl;
  std::pair<old::ul, old::ul> mid; old::ul m_dist;
  std::tie(mid, m_dist) = old::findTumor(view);
  std::pair<old::ul, old::ul> ori; std::pair<old::ul, old::ul> sub_dims;
  std::tie(ori, sub_dims) = old::findSubLattice(mid, m_dist, view);
  old::batchDiffuse(state.view().data, dims, params, 1);
  auto border = old::findBorderSites(sub_dims.first + 4, sub_dims.second+4, m_dist);
  for (int32_t r = 0; r < roi[0].dims.height+2; ++r) {
    for (int32_t c = 0; c < roi[0].dims.width+2; ++c) {
      std::cout << std::setw(4) << (int)(10*view(r + roi[0].origin.r - 1, c + roi[0].origin.c - 1).substrates.cho) << " ";
    }
    std::cout << std::endl;
  }
  ASSERT_EQ(roi[0].origin.r, ori.first);
  ASSERT_EQ(roi[0].origin.c, ori.second);
  ASSERT_EQ(roi[0].dims.height, sub_dims.first);
  ASSERT_EQ(roi[0].dims.width, sub_dims.second);
}

}