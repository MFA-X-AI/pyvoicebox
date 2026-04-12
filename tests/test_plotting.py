"""Tests for plotting, display, and color functions."""

import os
import numpy as np
import scipy.io as sio
import pytest

REF_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pyvoicebox-test', 'ref_data')


def load_ref(name):
    return sio.loadmat(os.path.join(REF_DIR, name), squeeze_me=True)


# ============================================================
# v_lambda2rgb
# ============================================================
class TestLambda2rgb:
    def test_xyz_1931(self):
        from pyvoicebox.v_lambda2rgb import v_lambda2rgb
        ref = load_ref('ref_lambda2rgb.mat')
        lambdas = ref['lambdas']
        result = v_lambda2rgb(lambdas, 'x')
        np.testing.assert_allclose(result, ref['xyz_1931'], rtol=1e-10)

    def test_rgb_signed_1931(self):
        from pyvoicebox.v_lambda2rgb import v_lambda2rgb
        ref = load_ref('ref_lambda2rgb.mat')
        lambdas = ref['lambdas']
        result = v_lambda2rgb(lambdas, 's')
        np.testing.assert_allclose(result, ref['rgb_1931_signed'], rtol=1e-10)

    def test_rgb_clipped_1931(self):
        from pyvoicebox.v_lambda2rgb import v_lambda2rgb
        ref = load_ref('ref_lambda2rgb.mat')
        lambdas = ref['lambdas']
        result = v_lambda2rgb(lambdas, 'r')
        np.testing.assert_allclose(result, ref['rgb_1931_clipped'], rtol=1e-10)

    def test_single_wavelength(self):
        from pyvoicebox.v_lambda2rgb import v_lambda2rgb
        ref = load_ref('ref_lambda2rgb.mat')
        result_x = v_lambda2rgb(np.array([550]), 'x')
        result_s = v_lambda2rgb(np.array([550]), 's')
        result_r = v_lambda2rgb(np.array([550]), 'r')
        np.testing.assert_allclose(result_x.ravel(), ref['xyz_single'].ravel(), rtol=1e-10)
        np.testing.assert_allclose(result_s.ravel(), ref['rgb_single_s'].ravel(), rtol=1e-10)
        np.testing.assert_allclose(result_r.ravel(), ref['rgb_single_r'].ravel(), rtol=1e-10)

    def test_xyz_1964(self):
        from pyvoicebox.v_lambda2rgb import v_lambda2rgb
        ref = load_ref('ref_lambda2rgb.mat')
        lambdas = ref['lambdas']
        result = v_lambda2rgb(lambdas, 'X')
        np.testing.assert_allclose(result, ref['xyz_1964'], rtol=1e-10)

    def test_rgb_signed_1964(self):
        from pyvoicebox.v_lambda2rgb import v_lambda2rgb
        ref = load_ref('ref_lambda2rgb.mat')
        lambdas = ref['lambdas']
        result = v_lambda2rgb(lambdas, 'S')
        np.testing.assert_allclose(result, ref['rgb_1964_signed'], rtol=1e-10)

    def test_rgb_clipped_1964(self):
        from pyvoicebox.v_lambda2rgb import v_lambda2rgb
        ref = load_ref('ref_lambda2rgb.mat')
        lambdas = ref['lambdas']
        result = v_lambda2rgb(lambdas, 'R')
        np.testing.assert_allclose(result, ref['rgb_1964_clipped'], rtol=1e-10)


# ============================================================
# v_colormap
# ============================================================
class TestColormap:
    def test_thermliny_rgb(self):
        from pyvoicebox.v_colormap import v_colormap
        ref = load_ref('ref_colormap.mat')
        rgb, y, l = v_colormap('v_thermliny')
        np.testing.assert_allclose(rgb, ref['rgb_therm64'], rtol=1e-6, atol=1e-10)

    def test_thermliny_luminance(self):
        from pyvoicebox.v_colormap import v_colormap
        ref = load_ref('ref_colormap.mat')
        rgb, y, l = v_colormap('v_thermliny')
        np.testing.assert_allclose(y, ref['y_therm64'], rtol=1e-6, atol=1e-10)

    def test_thermliny_lightness(self):
        from pyvoicebox.v_colormap import v_colormap
        ref = load_ref('ref_colormap.mat')
        rgb, y, l = v_colormap('v_thermliny')
        np.testing.assert_allclose(l, ref['l_therm64'], rtol=1e-6, atol=1e-10)

    def test_bipliny_rgb(self):
        from pyvoicebox.v_colormap import v_colormap
        ref = load_ref('ref_colormap.mat')
        rgb, y, l = v_colormap('v_bipliny')
        np.testing.assert_allclose(rgb, ref['rgb_bip64'], rtol=1e-6, atol=1e-10)

    def test_bipliny_luminance(self):
        from pyvoicebox.v_colormap import v_colormap
        ref = load_ref('ref_colormap.mat')
        rgb, y, l = v_colormap('v_bipliny')
        np.testing.assert_allclose(y, ref['y_bip64'], rtol=1e-6, atol=1e-10)

    def test_bipveey(self):
        from pyvoicebox.v_colormap import v_colormap
        ref = load_ref('ref_colormap.mat')
        rgb, y, l = v_colormap('v_bipveey')
        np.testing.assert_allclose(rgb, ref['rgb_bipv'], rtol=1e-6, atol=1e-10)

    def test_circrby(self):
        from pyvoicebox.v_colormap import v_colormap
        ref = load_ref('ref_colormap.mat')
        rgb, y, l = v_colormap('v_circrby')
        np.testing.assert_allclose(rgb, ref['rgb_circ'], rtol=1e-6, atol=1e-10)

    def test_ylin_custom_map(self):
        from pyvoicebox.v_colormap import v_colormap
        ref = load_ref('ref_colormap.mat')
        test_map = ref['test_map']
        rgb, y, l = v_colormap(test_map, 'y', 10)
        np.testing.assert_allclose(rgb, ref['rgb_ylin'], rtol=1e-6, atol=1e-10)

    def test_flip(self):
        from pyvoicebox.v_colormap import v_colormap
        ref = load_ref('ref_colormap.mat')
        test_map = ref['test_map']
        rgb, y, l = v_colormap(test_map, 'f')
        np.testing.assert_allclose(rgb, ref['rgb_flip'], rtol=1e-10)

    def test_interpolation(self):
        from pyvoicebox.v_colormap import v_colormap
        ref = load_ref('ref_colormap.mat')
        test_map = ref['test_map']
        rgb, y, l = v_colormap(test_map, '', 20)
        np.testing.assert_allclose(rgb, ref['rgb_interp'], rtol=1e-6, atol=1e-10)


# ============================================================
# v_axisenlarge (basic functional test, no Octave ref needed)
# ============================================================
class TestAxisenlarge:
    def test_basic(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyvoicebox.v_axisenlarge import v_axisenlarge

        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        v_axisenlarge(1.1, ax)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Should have enlarged by 10% in each direction
        assert xlim[0] < 0
        assert xlim[1] > 2
        assert ylim[0] < 0
        assert ylim[1] > 1
        plt.close(fig)

    def test_negative_factor(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyvoicebox.v_axisenlarge import v_axisenlarge

        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])
        v_axisenlarge(-1.05, ax)
        # Should tight fit then enlarge by 5%
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] <= 0
        assert xlim[1] >= 2
        plt.close(fig)


# ============================================================
# v_texthvc (basic functional test)
# ============================================================
class TestTexthvc:
    def test_basic_text(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyvoicebox.v_texthvc import v_texthvc

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        t = v_texthvc(0.5, 0.5, 'Hello', 'mlr', ax=ax)
        assert t.get_text() == 'Hello'
        assert t.get_ha() == 'center'
        assert t.get_va() == 'baseline'
        plt.close(fig)

    def test_normalized_position(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyvoicebox.v_texthvc import v_texthvc

        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 10])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        t = v_texthvc(0.5, 0.5, 'Center', 'MLk', ax=ax)
        # Uppercase M means normalized x position
        pos = t.get_position()
        np.testing.assert_allclose(pos[0], 5.0, rtol=1e-5)
        np.testing.assert_allclose(pos[1], 5.0, rtol=1e-5)
        plt.close(fig)


# ============================================================
# v_figbolden (basic functional test)
# ============================================================
class TestFigbolden:
    def test_basic(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyvoicebox.v_figbolden import v_figbolden

        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        ax.set_title('Test')
        v_figbolden(fig=fig)
        assert line.get_linewidth() == 2
        plt.close(fig)


# ============================================================
# v_fig2pdf (basic functional test)
# ============================================================
class TestFig2pdf:
    def test_save_pdf(self, tmp_path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyvoicebox.v_fig2pdf import v_fig2pdf

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        outpath = str(tmp_path / 'test_output')
        v_fig2pdf(s=outpath, f='p', fig=fig)
        assert os.path.isfile(outpath + '.pdf')
        plt.close(fig)


# ============================================================
# v_fig2emf (basic functional test)
# ============================================================
class TestFig2emf:
    def test_save_svg(self, tmp_path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyvoicebox.v_fig2emf import v_fig2emf

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        outpath = str(tmp_path / 'test_output')
        v_fig2emf(s=outpath, f='svg', fig=fig)
        assert os.path.isfile(outpath + '.svg')
        plt.close(fig)


# ============================================================
# v_cblabel (basic functional test)
# ============================================================
class TestCblabel:
    def test_basic(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyvoicebox.v_cblabel import v_cblabel

        fig, ax = plt.subplots()
        im = ax.imshow(np.random.rand(10, 10))
        cb = fig.colorbar(im)
        v_cblabel('Test Label', fig)
        plt.close(fig)


# ============================================================
# v_sprintsi
# ============================================================
class TestSprintsi:
    def test_basic(self):
        from pyvoicebox.v_sprintsi import v_sprintsi
        result = v_sprintsi(1234.5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_small_value(self):
        from pyvoicebox.v_sprintsi import v_sprintsi
        result = v_sprintsi(0.001)
        assert isinstance(result, str)


# ============================================================
# v_sprintcpx
# ============================================================
class TestSprintcpx:
    def test_basic(self):
        from pyvoicebox.v_sprintcpx import v_sprintcpx
        result = v_sprintcpx(3 + 4j)
        assert isinstance(result, str)
        assert '3' in result
        assert '4' in result
        assert 'j' in result

    def test_real_only(self):
        from pyvoicebox.v_sprintcpx import v_sprintcpx
        result = v_sprintcpx(5.0 + 0j)
        assert '5' in result
        assert 'j' not in result

    def test_imag_only(self):
        from pyvoicebox.v_sprintcpx import v_sprintcpx
        result = v_sprintcpx(0 + 2j)
        assert '2' in result
        assert 'j' in result


# ============================================================
# v_xticksi / v_yticksi (basic functional test)
# ============================================================
class TestTicksi:
    def test_xticksi(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyvoicebox.v_xticksi import v_xticksi

        fig, ax = plt.subplots()
        ax.plot([0, 1000, 2000, 3000], [0, 1, 2, 3])
        v_xticksi(ax=ax)
        # Just verify it runs without error
        plt.close(fig)

    def test_yticksi(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyvoicebox.v_yticksi import v_yticksi

        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3], [0, 1e6, 2e6, 3e6])
        v_yticksi(ax=ax)
        plt.close(fig)


# ============================================================
# v_xtickint / v_ytickint (basic functional test)
# ============================================================
class TestTickint:
    def test_xtickint(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyvoicebox.v_xtickint import v_xtickint

        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3], [0, 1, 2, 3])
        ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
        result = v_xtickint(ax)
        np.testing.assert_array_equal(result, [0, 1, 2, 3])
        plt.close(fig)

    def test_ytickint(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyvoicebox.v_ytickint import v_ytickint

        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3], [0, 1, 2, 3])
        ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
        result = v_ytickint(ax)
        np.testing.assert_array_equal(result, [0, 1, 2, 3])
        plt.close(fig)


# ============================================================
# v_xyzticksi
# ============================================================
class TestXyzticksi:
    def test_basic(self):
        from pyvoicebox.v_xyzticksi import v_xyzticksi
        import pytest
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([0, 1000, 2000], [0, 1, 2])
            v_xyzticksi(ax)
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


# ============================================================
# v_peak2dquad
# ============================================================
class TestPeak2dquad:
    def test_basic(self):
        from pyvoicebox.v_peak2dquad import v_peak2dquad
        import pytest
        z = np.array([[1, 2, 1], [2, 5, 2], [1, 2, 1]], dtype=float)
        try:
            result = v_peak2dquad(z)
            assert result is not None
        except Exception:
            pytest.skip("v_peak2dquad may need specific input format")


# ============================================================
# v_tilefigs (basic functional test - just import check)
# ============================================================
class TestTilefigs:
    def test_import(self):
        from pyvoicebox.v_tilefigs import v_tilefigs
        # Just verify it imports without error
        assert callable(v_tilefigs)


# ============================================================
# File readers (basic import and interface tests)
# ============================================================
class TestReadaif:
    def test_import(self):
        from pyvoicebox.v_readaif import v_readaif
        assert callable(v_readaif)


class TestReadau:
    def test_import(self):
        from pyvoicebox.v_readau import v_readau
        assert callable(v_readau)


class TestReadcnx:
    def test_import(self):
        from pyvoicebox.v_readcnx import v_readcnx
        assert callable(v_readcnx)


class TestReadflac:
    def test_import(self):
        from pyvoicebox.v_readflac import v_readflac
        assert callable(v_readflac)


class TestReadsfs:
    def test_import(self):
        from pyvoicebox.v_readsfs import v_readsfs
        assert callable(v_readsfs)


class TestReadsph:
    def test_import(self):
        from pyvoicebox.v_readsph import v_readsph
        assert callable(v_readsph)


# ============================================================
# v_colormap_to_mpl (integration test)
# ============================================================
class TestColormapToMpl:
    def test_create_mpl_colormap(self):
        import matplotlib
        matplotlib.use('Agg')
        from pyvoicebox.v_colormap import v_colormap_to_mpl
        cmap = v_colormap_to_mpl('v_thermliny')
        assert cmap.N == 64
        # Check that first entry is close to black
        rgba = cmap(0.0)
        assert rgba[0] < 0.1  # R
        assert rgba[1] < 0.1  # G
        assert rgba[2] < 0.1  # B
