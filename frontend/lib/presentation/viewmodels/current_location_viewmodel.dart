import 'package:frontend/data/repositories/current_location_repository_impl.dart';
import 'package:frontend/domain/entities/location.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'current_location_viewmodel.g.dart';

@Riverpod(keepAlive: true)
class CurrentLocationViewmodel extends _$CurrentLocationViewmodel {
  @override
  FutureOr<Location> build() async {
    return _loadCurrentLocation();
  }

  Future<Location> _loadCurrentLocation() async {
    return await ref
        .read(currentLocationRepositoryProvider)
        .getCurrentLocation();
  }
}
